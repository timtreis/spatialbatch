"""SequentialSampler -- ordered chunk-based sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.samplers import SequentialSampler as _AnnbatchSequentialSampler

from ._rows import RowsMixin

if TYPE_CHECKING:
    from ._rows import Rows


class SequentialSampler(RowsMixin, _AnnbatchSequentialSampler):
    """:class:`annbatch.samplers.SequentialSampler`, optionally restricted to `rows`.

    Parameters
    ----------
    chunk_size
        Number of consecutive cells per chunk.
    preload_nchunks
        Number of chunks to load per iteration.
    batch_size
        Number of observations per batch.
    drop_last
        Whether to drop the last incomplete batch.
    mask
        A slice defining the observation range to sample from (start:stop).
    rows
        The rows to sample from instead of `mask` (see :class:`~spatialbatch.samplers.RowsMixin`).
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        drop_last: bool = False,
        mask: slice | None = None,
        rows: Rows | None = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            drop_last=drop_last,
            mask=mask,
        )
        self._init_rows(rows)
