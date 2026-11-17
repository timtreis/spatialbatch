from __future__ import annotations

import asyncio
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
import zarr
import zarr.core.sync as zsync
from annbatch import Loader as AnnbatchLoader
from annbatch.utils import CSRContainer, check_lt_1, convert, load_all_aligned
from spatialdata import bounding_box_query
from spatialdata import rasterize as spatialdata_rasterize
from xarray import DataArray, DataTree

from spatialbatch.samplers import RandomSampler, SequentialSampler

from .utils import (
    TileGeometry,
    aligned_tiles,
    element_to_cs_matrix,
    is_identity_xy,
    scale0,
    tile_starts,
    validate_inputs,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from spatialdata import SpatialData

    from spatialbatch.io import SpatialDatasetCollection
    from spatialbatch.types import SpatialLoaderOutput

__all__ = ["Loader"]

# Side of the grid used to group the tiles of one request into windows that are each read from disk once. It is the
# default `block_px` of the collection writers, which sort the rows along a Z-curve over this grid of tile centres
# (`morton_order` in utils.py, called in io.py), so a run of consecutive rows falls in few windows.
_WINDOW_PX = 1024

_BATCHED_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"
_FUSED_PIPELINE = "zarr.core.codec_pipeline.FusedCodecPipeline"


def _fused(array: zarr.Array) -> zarr.Array:
    """`array`, reading through zarr's ``FusedCodecPipeline`` if this zarr has it and no other pipeline is configured.

    zarr's default ``BatchedCodecPipeline`` schedules one coroutine, thread hop and file read per inner chunk of a
    shard, and that Python work, under the GIL, is most of the CPU a tile read costs. ``FusedCodecPipeline`` (opt-in
    in zarr 3.4.2, ``zarr/core/config.py:113-123``; its docstring at ``zarr/core/codec_pipeline.py:973-997``) decodes
    the inner chunks of a shard synchronously from coalesced byte ranges (``ShardingCodec._decode_partial_sync``,
    ``zarr/codecs/sharding.py:1288-1375``), with one thread hop per read.

    The pipeline is fixed when an array is constructed (``AsyncArray.__init__`` calls ``create_codec_pipeline``,
    ``zarr/core/array.py:428-434``, which reads ``codec_pipeline.path`` through ``get_pipeline_class``,
    ``zarr/registry.py:458-468``), so setting the path only while this copy is built (``with_config``,
    ``zarr/core/array.py:1218-1239``) scopes it to the tile arrays: ``X`` and every other array keep zarr's
    configured pipeline. Pickling keeps the pipeline (``AsyncArray`` has no ``__getstate__``), so spawned DataLoader
    workers read through it too. A pipeline the user configured (e.g. zarrs) is left alone.
    """
    import zarr.core.codec_pipeline

    has_fused = hasattr(zarr.core.codec_pipeline, "FusedCodecPipeline")  # zarr >= 3.3
    if not has_fused or zarr.config.get("codec_pipeline.path") != _BATCHED_PIPELINE:
        return array
    with zarr.config.set({"codec_pipeline.path": _FUSED_PIPELINE}):
        return array.with_config({})


@dataclass
class _Source:
    """One pixel modality (an image or a labels element) of one SpatialData object."""

    kind: Literal["images", "labels"]
    array: zarr.Array | None  # scale 0; ``None`` when tiles must be produced by `fallback`
    channels: int | None  # ``None`` for labels
    dtype: np.dtype
    fallback: Callable[[int], np.ndarray] | None = None

    @property
    def signature(self) -> tuple[str, int | None, np.dtype]:
        return self.kind, self.channels, self.dtype


@dataclass
class _Item:
    """Everything the loader keeps about one SpatialData object besides its table ``X`` / ``obs`` / ``var``."""

    geometry: TileGeometry
    sources: dict[str, _Source]
    coords: np.ndarray | None


class _SpatialdataTile:
    """Produce one tile with :func:`spatialdata.rasterize` / :func:`spatialdata.bounding_box_query`, as ``XeniumDataset`` does.

    This is the path for elements whose pixel grid is not the coordinate system (a rotated H&E image, say) or when
    resampling is requested: correct for every transformation, but one spatialdata call per tile.
    A class rather than a closure so that the loader can be pickled into :class:`torch.utils.data.DataLoader` workers.
    """

    def __init__(
        self,
        element: DataArray | DataTree,
        bounds: np.ndarray,
        cs: str,
        rasterize: bool,
        rasterize_kwargs: Mapping[str, Any],
    ):
        self.element, self.bounds, self.cs = element, bounds, cs
        self.rasterize, self.rasterize_kwargs = rasterize, dict(rasterize_kwargs)

    def __call__(self, row: int) -> np.ndarray:
        minx, miny, maxx, maxy = self.bounds[row]
        kwargs = {
            "axes": ("x", "y"),
            "min_coordinate": np.array([minx, miny]),
            "max_coordinate": np.array([maxx, maxy]),
            "target_coordinate_system": self.cs,
        }
        tile = (
            spatialdata_rasterize(self.element, **kwargs, **self.rasterize_kwargs)
            if self.rasterize
            else bounding_box_query(self.element, **kwargs)
        )
        if isinstance(tile, DataTree):
            tile = next(iter(tile["scale0"].ds.values()))
        return np.asarray(tile.values)


class Loader(AnnbatchLoader):
    """A loader that streams multi-modal cell tiles from on-disk :class:`spatialdata.SpatialData` stores.

    This is :class:`annbatch.Loader` with pixels added. A *row* is a cell. For every row the loader yields the same
    things annbatch does (its ``X`` expression, ``obs`` and ``index``) plus a tile of every requested image and labels
    element, centered on the cell, so that all modalities of a batch are aligned along axis 0.

    It follows annbatch's recipe. A :class:`~annbatch.abc.Sampler` emits :class:`~annbatch.types.LoadRequest` s, each a
    set of chunks of consecutive rows; one request is read from disk **at once**, concurrently for ``X`` and for every
    modality, through async :mod:`zarr` (and `zarrs` if configured, see :doc:`zarr-configuration`); the in-memory
    result is shuffled and split into batches. No torch :class:`~torch.utils.data.DataLoader` is required, and when one
    is used it must be given ``batch_size=None``.

    Tiles that are a plain slice of the stored array (images in the pixel grid of the coordinate system, which is the
    case for Xenium ``morphology_focus``, ``morphology_mip`` and the labels) are read as a few large windows and
    cropped in memory. This yields exactly the pixels :func:`spatialdata.rasterize` returns (verified against
    ``XeniumDataset``), without resampling. Other elements, such as a rotated H&E image, are tiled with spatialdata.
    Where a tile crosses the border of the array the missing pixels are `pad_value`.

    Parameters
    ----------
    images
        Names of the image elements to tile.
    labels
        Names of the labels elements to tile.
    tile_scale
        Tile size as a multiple of the region's diameter, ignored if `tile_dim_in_units` is set.
        See :class:`spatialstream.dataloaders.XeniumDataset`.
    tile_dim_in_units
        Side of the tile in units of the coordinate system. All tiles of a loader must have the same integer size.
    crop_px
        Side in pixels of the tiles cut from a collection written by :func:`~spatialbatch.write_collection_dataset`,
        centred on ``obsm["spatial"]``. Required for such collections, ignored with a warning for collections written
        by :func:`~spatialbatch.write_sharded_sdata`, whose tiles are fixed. Unused by :meth:`add_sdata`.
    rasterize
        Mirror of ``XeniumDataset``: whether tiles are produced with :func:`spatialdata.rasterize` (``True``) or
        :func:`spatialdata.bounding_box_query` (``False``). It only matters on the spatialdata path.
    rasterize_kwargs
        Keyword arguments for :func:`spatialdata.rasterize`. The fast path requires ``{"target_unit_to_pixels": 1.0}``
        if `rasterize` is ``True``.
    pad_value
        Value of pixels outside the array.
    return_annotations
        `obs` columns to keep and yield. By default all are. A collection's datasets read only these columns of
        `obs` (none for ``[]``): `obs` is the largest part of a dataset held in memory, and reading it is a large part
        of :meth:`use_collection`.
    transform
        Called on every :class:`~spatialbatch.types.SpatialLoaderOutput` before it is yielded.
    batch_sampler
        If not provided, a :class:`~spatialbatch.samplers.RandomSampler` (if `shuffle`) or
        :class:`~spatialbatch.samplers.SequentialSampler` is built from the arguments below.
    chunk_size
        Number of consecutive cells per chunk. Mutually exclusive with `batch_sampler`. Defaults to 64.
    preload_nchunks
        Number of chunks per request. Mutually exclusive with `batch_sampler`. Defaults to 8.
    shuffle
        Whether to shuffle. Mutually exclusive with `batch_sampler`. Defaults to ``False``.
    batch_size
        Batch size to yield. Mutually exclusive with `batch_sampler`. Defaults to 32.
    drop_last
        Whether to drop the last incomplete batch. Mutually exclusive with `batch_sampler`. Defaults to ``False``.
    rng
        Random number generator for shuffling. Mutually exclusive with `batch_sampler`.
    return_index
        Whether to yield the position of each row in the loader: its global row, the datasets' rows in the order they
        were added (dataset ``i`` starts at the sum of the earlier datasets' ``n_obs``).
    rows
        Sample only these rows of the loader: a slice, a sequence of slices, an integer index array or a boolean
        mask over the global rows (see `return_index`). Chunks are drawn inside them, and a contiguous range is read
        as contiguous chunks. Mutually exclusive with `batch_sampler`.
    genes
        Yield only these columns of `X`, dense ``float32`` ``(batch, len(genes))`` in the order given, and the matching
        rows of `var`. A name is looked up in the var index (the Ensembl id in an sddb collection), then in
        ``var["original_var_name"]`` (the source name, e.g. the gene symbol) of a collection that has it. ``None``
        yields all of `X`, sparse.
    to
        Output library for `X` and for the tiles.
    tile_threads
        Threads per process that decode the tiles of a read, through zarr's ``FusedCodecPipeline`` (zarr >= 3.3),
        which reads the tiles with much less CPU than zarr's default pipeline. ``None`` reads them through the
        pipeline zarr is configured with, as does a zarr without it or a configured pipeline other than the default.
        Only arrays tiled in place are affected, not `X`.

    Build a loader once and iterate it many times: opening a collection reads every dataset's ``obs`` and checks its
    contract (see :meth:`use_collection`). With a torch :class:`~torch.utils.data.DataLoader`, use :meth:`to_torch`.

    Examples
    --------
    >>> from spatialbatch import Loader
    >>> loader = Loader(
    ...     images=["morphology_focus"],
    ...     labels=["cell_labels"],
    ...     tile_dim_in_units=224,
    ...     rasterize=True,
    ...     rasterize_kwargs={"target_unit_to_pixels": 1.0},
    ...     batch_size=32,
    ...     shuffle=True,
    ... ).add_sdata(sdata)
    >>> for batch in loader:
    ...     batch["images"]["morphology_focus"]  # (32, 1, 224, 224)
    """

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 64,
        "preload_nchunks": 8,
        "batch_size": 32,
        "drop_last": False,
    }

    _items: list[_Item]
    _signature: dict[str, tuple[str, int | None, np.dtype]] | None = None
    _tile_hw: tuple[int, int] | None = None

    def __init__(
        self,
        *,
        images: Sequence[str] = ("morphology_focus",),
        labels: Sequence[str] = (),
        tile_scale: float = 1.0,
        tile_dim_in_units: float | None = None,
        crop_px: int | None = None,
        rasterize: bool = False,
        rasterize_kwargs: Mapping[str, Any] = MappingProxyType({}),
        pad_value: float = 0,
        return_annotations: str | Sequence[str] | None = None,
        transform: Callable[[SpatialLoaderOutput], Any] | None = None,
        batch_sampler: Any = None,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        shuffle: bool | None = None,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        rng: np.random.Generator | None = None,
        return_index: bool = False,
        rows: slice | Sequence[slice] | np.ndarray | None = None,
        genes: Sequence[str] | None = None,
        to: Literal["torch", "jax"] | None = None,
        tile_threads: int | None = 4,
    ):
        # whether to_torch may set drop_last: only on a sampler built here, where the caller left it unset
        self._drop_last_settable = batch_sampler is None and drop_last is None
        if batch_sampler is None:
            core = {
                "chunk_size": chunk_size,
                "preload_nchunks": preload_nchunks,
                "batch_size": batch_size,
                "drop_last": drop_last,
            }
            resolved = {k: Loader._COMMON_SAMPLER_ARGS[k] if v is None else v for k, v in core.items()}
            if shuffle:
                batch_sampler = RandomSampler(**resolved, rng=rng, rows=rows)
            else:
                batch_sampler = SequentialSampler(**resolved, rows=rows)
            super().__init__(batch_sampler=batch_sampler, return_index=return_index, preload_to_gpu=False, to=to)
        else:  # let annbatch reject sampler arguments that conflict with a custom sampler
            if rows is not None:
                raise ValueError("rows is a sampler argument: pass it to the batch_sampler instead")
            super().__init__(
                batch_sampler=batch_sampler,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_last=drop_last,
                rng=rng,
                return_index=return_index,
                preload_to_gpu=False,
                to=to,
            )
        if not images and not labels:
            raise ValueError("At least one of `images` and `labels` must be given.")
        self._image_names, self._label_names = tuple(images), tuple(labels)
        if crop_px is not None and (isinstance(crop_px, bool) or not isinstance(crop_px, int) or crop_px < 1):
            raise ValueError(f"`crop_px` must be a positive integer number of pixels, got {crop_px}.")
        self._tile_scale, self._tile_dim_in_units, self._crop_px = tile_scale, tile_dim_in_units, crop_px
        if tile_threads is not None and (
            isinstance(tile_threads, bool) or not isinstance(tile_threads, int) or tile_threads < 1
        ):
            raise ValueError(f"`tile_threads` must be a positive integer or None, got {tile_threads}.")
        self._tile_threads = tile_threads
        self._rasterize, self._rasterize_kwargs = rasterize, dict(rasterize_kwargs)
        self._pad_value = pad_value
        self._annotations = [return_annotations] if isinstance(return_annotations, str) else return_annotations
        self.transform = transform
        self._items = []
        if genes is not None and (isinstance(genes, str) or len(genes) == 0):
            raise ValueError("`genes` must be a non-empty sequence of names")
        self._genes = None if genes is None else list(genes)
        self._gene_columns: np.ndarray | None = None
        self._shared_rows: Any = None  # a shared torch bool mask over the rows, set by to_torch for its workers
        self._shared_evaluate: Any = None  # a shared torch bool flag: the workers' pass is an evaluation (set_rows)
        self._full_var: pd.DataFrame | None = None  # a collection's var with its columns, for `genes` and batch var

    # -- adding data ---------------------------------------------------------------------------------------------

    def add_sdata(
        self,
        sdata: SpatialData,
        *,
        region: str = "cell_circles",
        table_name: str = "table",
        cs: str = "global",
    ) -> Self:
        """Append the cells of a backed :class:`spatialdata.SpatialData` to this loader.

        Parameters
        ----------
        sdata
            A SpatialData read from zarr (``sdata.is_backed()``), whose table annotates only `region`.
            The table's rows are the loader's rows, in table order.
        region
            The shapes (or labels) element whose centroids are the tile centers.
        table_name
            The table with the expression (``X``), ``obs`` and ``obsm["spatial"]``.
        cs
            The coordinate system the tiles are cut in.
        """
        if self._num_open_iters > 0:
            raise RuntimeError(
                "Cannot add datasets while the loader is being iterated. Finish or close the iterator first."
            )
        if not sdata.is_backed():
            raise ValueError("`sdata` must be backed by a zarr store (use `spatialdata.read_zarr`) to be streamed.")
        validate_inputs(sdata, region, self._image_names, self._label_names, cs, table_name)

        _, tiles = aligned_tiles(
            sdata, region, table_name, cs, tile_scale=self._tile_scale, tile_dim_in_units=self._tile_dim_in_units
        )
        geometry = tile_starts(tiles)
        bounds = tiles[["minx", "miny", "maxx", "maxy"]].to_numpy()

        sources = self._open_sources(sdata, bounds, geometry, cs)
        group = zarr.open_group(sdata.path, mode="r")["tables"][table_name]
        adata = load_all_aligned(group)
        obs = adata.obs if self._annotations is None else adata.obs[self._annotations]
        coords = np.asarray(adata.obsm["spatial"][...]) if "spatial" in adata.obsm else None
        self._add_item(adata.X, None if len(obs.columns) == 0 else obs, adata.var, _Item(geometry, sources, coords))
        return self

    def _open_sources(
        self, sdata: SpatialData, bounds: np.ndarray, geometry: TileGeometry, cs: str
    ) -> dict[str, _Source]:
        sources: dict[str, _Source] = {}
        native = (not self._rasterize) or self._rasterize_kwargs == {"target_unit_to_pixels": 1.0}
        for kind, names in (("images", self._image_names), ("labels", self._label_names)):
            for name in names:
                element = sdata[name]
                array = scale0(str(sdata.path), kind, name)
                if array.ndim != (3 if kind == "images" else 2):
                    raise NotImplementedError(f"Only 2D {kind} are supported, but `{name}` has shape {array.shape}.")
                channels = array.shape[0] if kind == "images" else None
                if native and is_identity_xy(element_to_cs_matrix(element, cs)):
                    sources[name] = _Source(kind, array, channels, array.dtype)
                else:
                    fallback = _SpatialdataTile(element, bounds, cs, self._rasterize, self._rasterize_kwargs)
                    probe = fallback(0)
                    expected = (geometry.height, geometry.width)
                    if probe.shape[-2:] != expected:
                        raise ValueError(
                            f"`{name}` yields tiles of shape {probe.shape} but {expected} pixels were expected; pass "
                            "`rasterize_kwargs={'target_unit_to_pixels': ...}` so that tiles are the same size."
                        )
                    sources[name] = _Source(kind, None, channels, probe.dtype, fallback)
        return sources

    def _add_item(self, X: Any, obs: pd.DataFrame | None, var: pd.DataFrame | None, item: _Item) -> None:
        """Add a dataset together with its pixel sources, atomically."""
        if self._tile_threads is not None:
            for source in item.sources.values():
                if source.array is not None:
                    source.array = _fused(source.array)
        signature = {name: src.signature for name, src in item.sources.items()}
        hw = (item.geometry.height, item.geometry.width)
        if self._items:
            if signature != self._signature:
                raise ValueError(
                    f"All datasets must provide the same modalities: expected {self._signature}, got {signature}."
                )
            if hw != self._tile_hw:
                raise ValueError(f"All datasets must have the same tile size: expected {self._tile_hw}, got {hw}.")
            if (item.coords is None) != (self._items[0].coords is None):
                raise ValueError("Either all or none of the datasets must have `obsm['spatial']`.")
        self._add_dataset_unchecked(X, obs, var)
        self._signature, self._tile_hw = signature, hw
        self._items.append(item)

    def add_sdatas(self, sdatas: Sequence[SpatialData], **kwargs: Any) -> Self:
        """Append several SpatialData objects, see :meth:`add_sdata`."""
        check_lt_1([len(sdatas)], ["Number of sdatas"])
        for sdata in sdatas:
            self.add_sdata(sdata, **kwargs)
        return self

    def use_collection(self, collection: SpatialDatasetCollection, **kwargs: Any) -> Self:  # type: ignore[override]
        """Load from a :class:`spatialbatch.SpatialDatasetCollection`. Can only be called once.

        A dataset that declares an sddb contract (``uns["sddb_contract"]``) is validated with
        ``sddb.contract.validate_dataset`` first, and refused if it does not conform. Without :mod:`sddb` installed,
        only the contract version is checked. A collection that passed is not validated again: the pass is recorded
        under a hash of the collection's consolidated metadata, its datasets' contracts and the sddb version, in this
        process and in ``$SPATIALBATCH_CACHE_DIR`` (default ``~/.cache/spatialbatch``). A change to an array's data
        that keeps its metadata (shape, dtype, attributes) is not noticed.
        """
        from spatialbatch.io import _check_collection

        if collection.is_empty:
            raise ValueError("SpatialDatasetCollection is empty")
        if self._collection_added:
            raise RuntimeError("You should not add multiple collections; add them to one collection instead.")
        groups = list(collection)
        if not _check_collection(groups, collection._group):
            warnings.warn(
                "The collection declares an sddb contract but `sddb` is not installed, so only the contract version "
                "was checked. Install `sddb` to validate it.",
                UserWarning,
                stacklevel=2,
            )
        for group in groups:
            self._add_group(group)
        self._full_var = collection.var
        self._collection_added = True
        return self

    def _add_group(self, group: zarr.Group) -> None:
        from spatialbatch.io import open_dataset_group

        X, obs, var, item = open_dataset_group(
            group, self._image_names, self._label_names, self._annotations, self._crop_px
        )
        self._add_item(X, obs, var, item)

    def _unsupported(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("A spatialbatch.Loader holds pixels too: use `add_sdata`, `add_sdatas` or `use_collection`.")

    add_adata = add_adatas = add_dataset = add_datasets = _unsupported  # type: ignore[assignment]

    def _resolve_genes(self) -> np.ndarray:
        """The columns of `X` that `genes` names, looked up once."""
        if self._gene_columns is not None:
            return self._gene_columns
        assert self._genes is not None
        var = self._full_var if self._full_var is not None else self._var
        if var is None:
            raise ValueError("`genes` needs the datasets' var, but the loader has none")
        index = pd.Index(var.index.astype(str))
        symbols = pd.Index(var["original_var_name"].astype(str)) if "original_var_name" in var.columns else None
        columns, missing = [], []
        for name in self._genes:
            if (hit := index.get_indexer([name])[0]) >= 0:
                columns.append(hit)
            elif symbols is not None and (hits := np.flatnonzero(symbols == name)).size == 1:
                columns.append(int(hits[0]))
            elif symbols is not None and hits.size > 1:
                raise ValueError(f"gene {name!r} is the original_var_name of {hits.size} features: pass its var name")
            else:
                missing.append(name)
        if missing:
            raise KeyError(f"genes not in the var index or var['original_var_name']: {missing[:10]}")
        self._gene_columns = np.asarray(columns, dtype=np.intp)
        return self._gene_columns

    def to_torch(
        self,
        num_workers: int = 0,
        *,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """A :class:`torch.utils.data.DataLoader` over this loader, set up the way it needs.

        - ``batch_size=None``: the loader makes the batches itself;
        - with more than one worker the sampler must drop the last incomplete batch, which annbatch requires
          (``annbatch/samplers/_chunk_sampler.py``, ``_validate_worker_mode``). If the loader built its sampler and
          `drop_last` was not given, it is set to ``True`` here: each worker then skips at most its last
          ``batch_size - 1`` rows. A sampler that was given, or ``drop_last=False``, raises instead. annbatch also
          refuses several workers for a sampler that does not shuffle. An evaluation pass (:meth:`set_rows` with
          ``evaluate=True``) needs neither: it yields every row once with any number of workers;
        - `pin_memory` defaults to whether CUDA is available, `persistent_workers` to ``num_workers > 0``.

        The DataLoader applies torch's ``default_convert`` to every batch, so the arrays (tiles, ``index``,
        ``dataset``, ``coords``, and `X` with `genes`) arrive as tensors with any number of workers; build the loader
        with ``to="torch"`` to get `X` as a torch sparse tensor too. Each worker holds its own copy of the loader.

        Parameters
        ----------
        num_workers
            Worker processes of the DataLoader.
        pin_memory, persistent_workers
            See :class:`torch.utils.data.DataLoader`.
        **kwargs
            Passed to :class:`torch.utils.data.DataLoader`, e.g. ``prefetch_factor``. `batch_size` must not be given.
        """
        import torch
        from torch.utils.data import DataLoader

        if "batch_size" in kwargs or "batch_sampler" in kwargs or "sampler" in kwargs:
            raise ValueError("the loader makes its own batches: do not pass batch_size, batch_sampler or sampler")
        if num_workers > 1 and not getattr(self._batch_sampler, "_evaluate", False):
            sampler = self._batch_sampler
            if not getattr(sampler, "shuffle", True):
                raise ValueError(
                    "annbatch does not support several workers with a sampler that does not shuffle; use shuffle=True "
                    "(with drop_last) or num_workers <= 1"
                )
            if not getattr(sampler, "_drop_last", True) and sampler.batch_size > 1:
                if not self._drop_last_settable:
                    raise ValueError(
                        "annbatch needs drop_last=True with several workers; build the Loader (or its sampler) with it"
                    )
                sampler._drop_last = True
        from spatialbatch.samplers._rows import RowsMixin

        if num_workers > 0 and isinstance(self._batch_sampler, RowsMixin):
            self._shared_rows = torch.from_numpy(self._rows_mask(self._batch_sampler._rows)).share_memory_()
            self._shared_evaluate = torch.tensor([self._batch_sampler._evaluate]).share_memory_()
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        return DataLoader(
            self,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    def set_rows(self, rows: slice | Sequence[slice] | np.ndarray | None, *, evaluate: bool = False) -> Self:
        """Sample `rows` (see the `rows` parameter; ``None`` for all) from the next iteration on.

        This lets one loader, and the workers of its :meth:`to_torch` DataLoader, serve several row sets, e.g. a
        training and a validation split, instead of one loader (and one set of workers, each with its own copy of the
        loader) per split. Persistent workers keep the loader they were started with, so :meth:`to_torch` gives them
        a shared row mask and evaluation flag, which this method updates and each worker reads when an epoch starts
        (``iter(dataloader)``). Call it between epochs, not while one is running.

        Parameters
        ----------
        rows
            The rows to sample, as the `rows` parameter of the loader.
        evaluate
            ``False``: the sampler keeps its settings, so with workers the rows are shuffled and each worker drops at
            most its last ``batch_size - 1`` of them (annbatch needs ``drop_last`` with several workers).
            ``True``: an evaluation pass, every row exactly once: in row order, split into one contiguous share per
            worker, nothing shuffled or dropped. Batches can then be shorter than ``batch_size``, and with workers
            they arrive interleaved across the shares. Use it for validation and inference.
        """
        from spatialbatch.samplers._rows import RowsMixin

        if self._num_open_iters > 0:
            raise RuntimeError("set_rows while the loader is being iterated; finish or close the iterator first")
        if not isinstance(self._batch_sampler, RowsMixin):
            raise TypeError("set_rows needs a spatialbatch sampler (RandomSampler or SequentialSampler)")
        mask = self._rows_mask(rows)  # checks `rows` now, not at the next epoch
        self._batch_sampler._rows = rows
        self._batch_sampler._evaluate = bool(evaluate)
        if self._shared_rows is not None:
            import torch

            self._shared_rows.copy_(torch.from_numpy(mask))
            self._shared_evaluate.fill_(bool(evaluate))
        return self

    def _rows_mask(self, rows: Any) -> np.ndarray:
        from spatialbatch.samplers._rows import resolve_rows

        mask = np.ones(self.n_obs, bool)
        if rows is not None:
            mask[:] = False
            mask[resolve_rows(rows, self.n_obs)] = True
        return mask

    def _rows_from_shared_mask(self) -> None:
        """In a DataLoader worker: take the rows that :meth:`set_rows` last put in the shared mask."""
        if self._shared_rows is None:
            return
        from annbatch.samplers._utils import get_torch_worker_info

        if get_torch_worker_info() is None:
            return
        mask = self._shared_rows.numpy().copy()
        self._batch_sampler._rows = None if mask.all() else mask
        self._batch_sampler._evaluate = bool(self._shared_evaluate[0])

    @property
    def n_tiles_px(self) -> tuple[int, int]:
        """Height and width in pixels of every tile."""
        if self._tile_hw is None:
            raise ValueError("No datasets added yet")
        return self._tile_hw

    # -- reading -------------------------------------------------------------------------------------------------

    async def _index_pixels(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> dict[str, np.ndarray]:
        """Fetch the tiles of the requested rows of every modality into preallocated buffers.

        The buffers have the row order of the ``X`` buffer of :meth:`_index_datasets`: grouped by dataset.
        """
        assert self._signature is not None and self._tile_hw is not None
        total = sum(len(rows) for rows in dataset_index_to_rows.values())
        th, tw = self._tile_hw
        buffers = {
            name: np.empty((total, *(() if channels is None else (channels,)), th, tw), dtype)
            for name, (_, channels, dtype) in self._signature.items()
        }
        tasks, offset = [], 0
        for dataset_idx, rows in dataset_index_to_rows.items():
            item = self._items[dataset_idx]
            for name, source in item.sources.items():
                out = buffers[name][offset : offset + len(rows)]
                if source.fallback is None:
                    tasks.append(self._fetch_tiles(source, item.geometry, rows, out))
                else:
                    tasks.append(self._fetch_tiles_fallback(source, rows, out))
            offset += len(rows)
        await asyncio.gather(*tasks)
        return buffers

    async def _fetch_tiles(self, source: _Source, geometry: TileGeometry, rows: np.ndarray, out: np.ndarray) -> None:
        """Fill `out` with the tiles of `rows`, reading each window of the array that holds several tiles only once."""
        assert source.array is not None
        th, tw = geometry.height, geometry.width
        height, width = source.array.shape[-2:]
        y0, x0 = geometry.y0[rows], geometry.x0[rows]
        cy0, cy1 = np.clip(y0, 0, height), np.clip(y0 + th, 0, height)
        cx0, cx1 = np.clip(x0, 0, width), np.clip(x0 + tw, 0, width)
        inside = (cy1 > cy0) & (cx1 > cx0)
        whole = inside & (cy1 - cy0 == th) & (cx1 - cx0 == tw)
        if not whole.all():  # only the tiles at the border
            out[~whole] = self._pad_value
        members = np.flatnonzero(inside)
        if members.size == 0:
            return
        # Group by the window that holds the tile's centre: the writers sort rows by centre on this grid, while the
        # clipped origins sit half a tile off it. A window reads the bounding box of its tiles, so a tile that
        # crosses the grid is still read whole.
        yc = np.clip(y0[members] + th // 2, 0, height - 1)
        xc = np.clip(x0[members] + tw // 2, 0, width - 1)
        cells = (yc // _WINDOW_PX) * (width // _WINDOW_PX + 1) + xc // _WINDOW_PX
        order = np.argsort(cells, kind="stable")
        members, cells = members[order], cells[order]
        bounds = np.flatnonzero(np.diff(cells)) + 1
        arr = source.array._async_array

        async def read_window(group: np.ndarray) -> None:
            wy0, wy1 = int(cy0[group].min()), int(cy1[group].max())
            wx0, wx1 = int(cx0[group].min()), int(cx1[group].max())
            selection = (slice(wy0, wy1), slice(wx0, wx1))
            window = await arr.getitem(selection if source.channels is None else (slice(None), *selection))
            for j, sy0, sy1, sx0, sx1, dy0, dx0 in zip(
                group.tolist(),
                (cy0[group] - wy0).tolist(),
                (cy1[group] - wy0).tolist(),
                (cx0[group] - wx0).tolist(),
                (cx1[group] - wx0).tolist(),
                (cy0[group] - y0[group]).tolist(),
                (cx0[group] - x0[group]).tolist(),
                strict=True,
            ):
                out[j, ..., dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = window[..., sy0:sy1, sx0:sx1]

        await asyncio.gather(*(read_window(g) for g in np.split(members, bounds)))

    async def _fetch_tiles_fallback(self, source: _Source, rows: np.ndarray, out: np.ndarray) -> None:
        """Fill `out` with one spatialdata call per tile, spread over threads."""
        assert source.fallback is not None
        n_threads = zarr.config.get("threading.max_workers") or os.cpu_count() or 1

        def work() -> None:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for j, tile in enumerate(pool.map(source.fallback, rows.tolist())):
                    out[j] = tile

        await asyncio.get_running_loop().run_in_executor(None, work)

    async def _index_all(
        self, dataset_index_to_rows: OrderedDict[int, np.ndarray]
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Fetch ``X`` and all pixel modalities concurrently."""
        x, pixels = await asyncio.gather(
            self._index_datasets(dataset_index_to_rows), self._index_pixels(dataset_index_to_rows)
        )
        return x, pixels

    def _concatenate_outs(self, outs: OrderedDict[int, Any]) -> Any:
        """Annbatch's concatenation, with each dataset's ``indptr`` first cast to the dtype of the output ``indptr``.

        When the datasets do not share a dtype, annbatch 0.3.1 allocates a dataset's ``indptr`` as
        ``np.min_scalar_type`` of that dataset's nnz in the batch (``Loader._allocate_per_dataset_outs``) and adds the
        running nnz offset to it (``Loader._concatenate_outs``). Under NumPy 2
        promotion a ``uint16`` ``indptr`` plus an offset above 65535 raises ``OverflowError``: a batch over several
        datasets fails when a later dataset adds fewer than 65536 nonzeros. Drop this once annbatch casts it.
        """
        values = list(outs.values())
        if values and isinstance(values[0], CSRContainer):
            dtype = np.min_scalar_type(sum(o.elems[0].size for o in values))
            for key, o in outs.items():
                data, indices, indptr = o.elems
                outs[key] = CSRContainer(
                    elems=(data, indices, indptr.astype(dtype, copy=False)), shape=o.shape, dtype=o.dtype
                )
        return super()._concatenate_outs(outs)

    def _maybe_accumulate_coords(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> np.ndarray | None:
        coords = [self._items[idx].coords for idx in dataset_index_to_rows]
        if any(c is None for c in coords):
            return None
        return np.concatenate([c[rows] for c, rows in zip(coords, dataset_index_to_rows.values(), strict=True)])

    def _iter(self) -> Iterator[SpatialLoaderOutput]:  # type: ignore[override]
        check_lt_1([len(self._train_datasets), self.n_obs], ["Number of datasets", "Number of observations"])
        is_sparse = not issubclass(self.dataset_type, zarr.Array)
        self._rows_from_shared_mask()
        positions = np.empty(0, dtype=np.intp)
        for load_request in self._batch_sampler.sample(self.n_obs):
            requests_to_load = load_request.get("requests", None)
            if requests_to_load is None:
                raise KeyError("load_request must contain either 'requests'.")
            splits = load_request["splits"]

            dataset_index_to_rows, order = self._requests_to_dataset_rows(requests_to_load)
            # See annbatch.Loader._iter: `inv` maps a position in the request to its position in the buffers.
            n = order.size
            inv_buffer = np.empty(n, dtype=np.intp)
            if n > positions.size:
                positions = np.arange(n, dtype=np.intp)
            inv = inv_buffer[:n]
            inv[order] = positions[:n]

            # The fused pipeline takes its thread count from the global config when it reads, not from the array
            # (`max_workers=_resolve_max_workers()` in `FusedCodecPipeline.read`, zarr/core/codec_pipeline.py:1317-1319
            # and :54-74 @ zarr 3.4.2), so it is set around the read. Only that pipeline reads this key
            # (zarr/core/config.py:122), so the X read is unaffected; the setting is process-global while it is set.
            threads = self._tile_threads
            with nullcontext() if threads is None else zarr.config.set({"codec_pipeline.max_workers": threads}):
                raw_out, pixels = zsync.sync(self._index_all(dataset_index_to_rows))

            if is_sparse:
                in_memory_data = self._sp_module.csr_matrix(
                    tuple(self._np_module.asarray(e) for e in raw_out.elems), shape=raw_out.shape, dtype=raw_out.dtype
                )
            else:
                in_memory_data = self._np_module.asarray(raw_out)

            genes = None
            if self._genes is not None:
                columns = self._resolve_genes()
                part = in_memory_data[:, columns]
                genes = np.asarray(part.toarray() if hasattr(part, "toarray") else part, dtype=np.float32)
                var = self._batch_var(columns)
            else:
                var = self._var
            obs = self._maybe_accumulate_obs(dataset_index_to_rows)
            index = self._maybe_accumulate_indices(dataset_index_to_rows)
            dataset = np.concatenate([np.full(len(rows), idx, np.int64) for idx, rows in dataset_index_to_rows.items()])
            coords = self._maybe_accumulate_coords(dataset_index_to_rows)
            for split in splits:
                sel = inv[split]
                data = in_memory_data[sel] if genes is None else genes[sel]
                batch: SpatialLoaderOutput = {
                    "X": data if self._to is None else convert(data, False, self._to),
                    "obs": obs.iloc[sel] if obs is not None else None,
                    "var": var,
                    "index": index[sel] if index is not None else None,
                    "dataset": dataset[sel],
                    "images": {name: self._convert_tiles(pixels[name][sel]) for name in self._image_names},
                    "labels": {name: self._convert_tiles(pixels[name][sel]) for name in self._label_names},
                    "coords": coords[sel] if coords is not None else None,
                }
                yield batch if self.transform is None else self.transform(batch)

    def _batch_var(self, columns: np.ndarray) -> pd.DataFrame | None:
        var = self._full_var if self._full_var is not None else self._var
        return None if var is None else var.iloc[columns]

    def _convert_tiles(self, tiles: np.ndarray) -> Any:
        return tiles if self._to is None else convert(tiles, False, self._to)
