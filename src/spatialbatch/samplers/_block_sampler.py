"""BlockSampler -- shuffled sampler whose chunks are the writer's spatial blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng

from ._random_sampler import RandomSampler
from ._rows import _runs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest

    from ._rows import Rows


class BlockSampler(RandomSampler):
    """Shuffled sampler whose chunks are whole blocks: the runs of rows that the collection writer put in one grid cell.

    :func:`~spatialbatch.write_collection_dataset` sorts a dataset's rows along a Z-curve over a ``block_px`` grid of
    the cell centres, so the rows of one grid cell are consecutive and their tiles lie in one window of the image. This
    sampler draws those blocks (about 30 rows each for Visium spots at 0.5 µm/px and 1024 px) in random order instead
    of fixed runs of `chunk_size` rows, so a request reads one window per block and still mixes many datasets. A block
    of more than `chunk_size` rows is cut into runs of `chunk_size` consecutive rows: at 5 µm/px a 1024 px cell holds
    most of a Visium slide, and whole blocks gave 1.9 datasets per batch instead of 7.9 with row chunks.

    The blocks, in random order, are cut into requests of ``chunk_size * preload_nchunks`` rows, as for
    :class:`RandomSampler`; a block that crosses the end of a request continues in the next one. Each request is
    shuffled in memory and split into batches. Worker sharding (each worker a share of the shuffled blocks), `rows`
    (blocks are cut to the selected rows) and `drop_last` (at most the last ``batch_size - 1`` rows of a pass, of each
    worker's pass with workers) work as in :class:`RandomSampler`.

    The loader gives the sampler the first row of every block (:attr:`block_starts`) before each pass: for a
    collection, the rows where the writer's grid cell changes, recomputed from ``obsm["spatial"]`` and ``block_px``;
    otherwise, where the cell of the tile centre on a 1024 px grid changes. Every dataset starts a block.

    Parameters
    ----------
    chunk_size, preload_nchunks
        A request holds ``chunk_size * preload_nchunks`` rows; it must be divisible by `batch_size`. A block holds at
        most `chunk_size` rows.
    batch_size
        Number of observations per batch.
    drop_last
        Whether to drop the last incomplete batch.
    rows
        The rows to sample from (see :class:`~spatialbatch.samplers.RowsMixin`).
    rng
        Random number generator for shuffling, a seeded :class:`numpy.random.Generator` for reproducibility.
    """

    #: The first global row of every block, sorted, starting with 0. Set by :class:`~spatialbatch.Loader`.
    block_starts: np.ndarray | None = None

    def __init__(
        self,
        chunk_size: int = 64,
        preload_nchunks: int = 8,
        batch_size: int = 32,
        *,
        drop_last: bool = False,
        rows: Rows | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(chunk_size, preload_nchunks, batch_size, drop_last=drop_last, rows=rows, rng=rng)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        if self._evaluate:  # every row once, in row order: the row chunks already read in order
            yield from self._sample_all_once(n_obs)
            return
        if self.block_starts is None:
            raise RuntimeError("BlockSampler needs the loader's block starts: use it through spatialbatch.Loader")
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)
        batch_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)
        positions = self._selected(n_obs)
        block = np.searchsorted(self.block_starts, positions, side="right")
        new_block = np.concatenate([[True], block[1:] != block[:-1]])
        # a block longer than chunk_size rows is cut into runs of chunk_size consecutive (Morton-ordered) rows: on a
        # low-resolution image one block_px cell can hold most of a slide, and a request would then mix few datasets
        nth = np.arange(positions.size)
        in_block = nth - np.maximum.accumulate(np.where(new_block, nth, 0))
        blocks = _runs(positions, in_block % self._chunk_size == 0)
        order = self._rng.permutation(len(blocks))  # the shared rng: every worker draws the same order
        if worker_info is not None:
            order = np.array_split(order, worker_info.num_workers)[worker_info.id]
        yield from self._requests(_pack([blocks[i] for i in order], self._in_memory_size), batch_rng)


def _pack(blocks: list[list[slice]], size: int) -> list[list[slice]]:
    """The slices of `blocks`, in order, cut into groups of `size` rows (the last may be shorter)."""
    groups: list[list[slice]] = [[]]
    room = size
    for s in (s for block in blocks for s in block):
        start = s.start
        while start < s.stop:
            if room == 0:
                groups.append([])
                room = size
            take = min(room, s.stop - start)
            groups[-1].append(slice(start, start + take))
            start, room = start + take, room - take
    return groups if groups[0] else []
