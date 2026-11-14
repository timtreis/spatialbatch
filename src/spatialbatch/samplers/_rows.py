"""Chunk sampling restricted to a subset of the loader's rows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from annbatch.types import LoadRequest

    Rows = slice | Sequence[slice] | np.ndarray

__all__ = ["RowsMixin", "resolve_rows"]


def resolve_rows(rows: Rows, n_obs: int) -> np.ndarray:
    """The sorted, unique global row positions that `rows` selects out of `n_obs`.

    `rows` is a slice, a sequence of slices, an integer index array or a boolean mask of length `n_obs`.
    """
    if isinstance(rows, slice):
        rows = [rows]
    if isinstance(rows, list | tuple) and all(isinstance(r, slice) for r in rows):
        parts = [np.arange(*r.indices(n_obs)) for r in rows]
        if any(r.step not in (None, 1) for r in rows):
            raise ValueError("rows: slices must have step 1")
        out = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
    else:
        arr = np.asarray(rows)
        if arr.dtype == bool:
            if arr.shape != (n_obs,):
                raise ValueError(f"rows: a boolean mask must have length n_obs ({n_obs}), got {arr.shape}")
            out = np.flatnonzero(arr)
        elif np.issubdtype(arr.dtype, np.integer):
            out = np.unique(arr)
            if out.size and (out[0] < 0 or out[-1] >= n_obs):
                raise ValueError(f"rows: indices must be in [0, {n_obs})")
        else:
            raise TypeError(f"rows must be a slice, slices, an integer array or a boolean mask; got {arr.dtype}")
    if out.size == 0:
        raise ValueError("rows selects no row")
    return out.astype(np.int64, copy=False)


def _chunks(positions: np.ndarray, chunk_size: int) -> list[list[slice]]:
    """`positions` cut into chunks of `chunk_size` rows (the last may be shorter), each as its contiguous runs."""
    i = np.arange(positions.size)
    breaks = np.flatnonzero((i % chunk_size == 0) | np.concatenate([[True], np.diff(positions) != 1]))
    ends = np.append(breaks[1:], positions.size)
    chunks: list[list[slice]] = [[] for _ in range(math.ceil(positions.size / chunk_size))]
    for b, e in zip(breaks, ends, strict=True):
        chunks[b // chunk_size].append(slice(int(positions[b]), int(positions[e - 1]) + 1))
    return chunks


class RowsMixin:
    """Restrict an annbatch chunk sampler to a subset of rows (``rows=``).

    annbatch's samplers take one contiguous ``mask`` slice (``annbatch.samplers._utils.validate_mask_and_resolve``).
    Here a *chunk* is `chunk_size` consecutive *selected* rows, given to the loader as the contiguous runs it spans, so
    a contiguous range is still read as contiguous slices. Chunks are shuffled (if the sampler shuffles) and grouped
    `preload_nchunks` to a request; every request but the last holds ``chunk_size * preload_nchunks`` rows, as in
    annbatch, and the shorter last chunk is always last. Worker sharding and the per-worker batch RNG follow
    ``annbatch.samplers._chunk_sampler._ChunkSampler._sample``. Sampling with replacement or ``num_samples`` is not
    supported together with `rows`.
    """

    _rows: Rows | None = None

    def _init_rows(self, rows: Rows | None) -> None:
        self._rows = rows
        if rows is not None and (getattr(self, "_replacement", False) or getattr(self, "_num_samples", None)):
            raise ValueError("rows cannot be combined with replacement or num_samples")
        if rows is not None and self._mask != slice(0, None):  # type: ignore[attr-defined]
            raise ValueError("pass either rows or mask, not both")

    def validate(self, n_obs: int) -> None:
        if self._rows is None:
            return super().validate(n_obs)  # type: ignore[misc]
        resolve_rows(self._rows, n_obs)
        return None

    def n_batches(self, n_obs: int) -> int:
        if self._rows is None:
            return super().n_batches(n_obs)  # type: ignore[misc]
        total = resolve_rows(self._rows, n_obs).size
        return total // self.batch_size if self._drop_last else math.ceil(total / self.batch_size)  # type: ignore[attr-defined]

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        if self._rows is None:
            yield from super()._sample(n_obs)  # type: ignore[misc]
            return
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)  # type: ignore[attr-defined]
        rng: np.random.Generator = self._rng  # type: ignore[attr-defined]
        batch_rng = rng if worker_info is None else _spawn_worker_rng(rng, worker_info.id)
        chunks = _chunks(resolve_rows(self._rows, n_obs), self._chunk_size)  # type: ignore[attr-defined]
        if self.shuffle:  # type: ignore[attr-defined]
            order = np.arange(len(chunks) - 1)
            rng.shuffle(order)  # the shared rng: every worker draws the same order and takes its own shard
            chunks = [chunks[i] for i in order] + [chunks[-1]]
        if worker_info is not None:
            shard = np.array_split(np.arange(len(chunks)), worker_info.num_workers)[worker_info.id]
            chunks = [chunks[i] for i in shard]
        yield from self._requests(chunks, batch_rng)

    def _requests(self, chunks: list[list[slice]], batch_rng: np.random.Generator) -> Iterator[LoadRequest]:
        per = self._preload_nchunks  # type: ignore[attr-defined]
        groups = [chunks[i : i + per] for i in range(0, len(chunks), per)]
        batch_size, shuffle = self.batch_size, self.shuffle  # type: ignore[attr-defined]
        for i, group in enumerate(groups):
            requests = [s for chunk in group for s in chunk]
            n = int(sum(s.stop - s.start for s in requests))
            if i == len(groups) - 1 and self._drop_last:  # type: ignore[attr-defined]
                if n < batch_size:
                    return
                n -= n % batch_size
            indices = batch_rng.permutation(n) if shuffle else np.arange(n)
            yield {"requests": requests, "splits": split_given_size(indices, batch_size)}
