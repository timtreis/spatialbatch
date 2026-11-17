from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import zarr
from annbatch.io import DATASET_PREFIX, V1_ENCODING, DatasetCollection, write_sharded
from tqdm.auto import tqdm
from zarr.codecs import ZstdCodec

from .utils import (
    TileGeometry,
    aligned_tiles,
    element_to_cs_matrix,
    is_identity_xy,
    morton_codes,
    morton_order,
    round_half_up,
    scale0,
    tile_starts,
    validate_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import ArrayLike
    from spatialdata import SpatialData
    from zarr.abc.codec import BytesBytesCodec

    from .loader import _Item

__all__ = ["SpatialDatasetCollection", "spatial_split", "write_collection_dataset", "write_sharded_sdata"]

#: Key of the group attribute that describes how a dataset of a collection was written.
ENCODING_KEY = "spatialbatch"
#: Datasets whose tiles are cut at read time, ``crop_px`` square around ``obsm["spatial"]``. Their ``centres``
#: attribute names that element; it is descriptive, the loader always reads ``obsm["spatial"]``.
CENTRES_VERSION = "0.2.0"
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$")


def _version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("-")[0].split("+")[0].split("."))


def _copy_sharded(
    src: Any,
    group: zarr.Group,
    name: str,
    *,
    chunk: int,
    shard: int,
    compressors: Iterable[BytesBytesCodec],
) -> None:
    """Copy `src` to a sharded array, one shard-sized block at a time so that memory use stays at ~one shard.

    `src` is a numpy, zarr or dask array. A dask array is rechunked to the shards and stored with
    :func:`dask.array.store`, so that it is computed block by block and no two tasks write to the same shard.
    """
    lead = src.shape[:-2]
    height, width = src.shape[-2:]
    chunks = (min(chunk, height), min(chunk, width))
    # a shard must be a multiple of the chunk, also when the array is smaller than one shard
    shards = tuple(min(shard, -(-side // c) * c) for side, c in zip((height, width), chunks, strict=True))
    dst = group.create_array(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=(*lead, *chunks),
        shards=(*lead, *shards),
        compressors=tuple(compressors),
        fill_value=0,
        overwrite=True,
    )
    if hasattr(src, "dask"):
        import dask.array as da

        da.store(src.rechunk(dst.shards), dst, lock=False)
        return
    blocks = [(y, x) for y in range(0, height, shard) for x in range(0, width, shard)]
    for y, x in tqdm(blocks, desc=f"Writing {name}", leave=False):
        dst[..., y : y + shard, x : x + shard] = np.asarray(src[..., y : y + shard, x : x + shard])


def _write_dataset(
    group: zarr.Group,
    adata: AnnData,
    *,
    key: str,
    images: Mapping[str, Any],
    labels: Mapping[str, Any],
    encoding: Mapping[str, Any],
    n_obs_per_chunk: int,
    shard_size: int | str,
    image_chunk: int,
    image_shard: int,
    compressors: Iterable[BytesBytesCodec],
) -> None:
    """Write an already sorted table and its pixel arrays at ``group[key]``, then the encoding attributes."""
    if image_shard % image_chunk != 0:
        raise ValueError(f"image_shard ({image_shard}) must be divisible by image_chunk ({image_chunk}).")
    write_sharded(group, adata, n_obs_per_chunk=min(n_obs_per_chunk, adata.shape[0]), shard_size=shard_size, key=key)
    out = group[key]
    for kind, arrays in (("images", images), ("labels", labels)):
        for name, array in arrays.items():
            _copy_sharded(
                array, out.require_group(kind), name, chunk=image_chunk, shard=image_shard, compressors=compressors
            )
    out.update_attributes({ENCODING_KEY: dict(encoding)})
    group.update_attributes(V1_ENCODING)


def _in_order(table: AnnData, order: np.ndarray) -> AnnData:
    """The rows of `table` in `order`, holding one X: the input's when `order` is the identity, else one reordered.

    `table` is left unchanged, and ``uns["spatialdata_attrs"]`` is left out, as it names the elements of the source.
    ``AnnData.copy`` would copy X as well ("Full copy", ``anndata/_core/anndata.py:1447-1448``), so X is taken apart
    from the rest. A dask X is computed one row block at a time straight into its new rows (:func:`_compute_in_order`);
    a backed sparse X (anndata's ``sparse_dataset``) reads its rows in `order` from the store
    (``anndata/_core/sparse_dataset.py:318-325, 403-406``). The other elements, such as layers and
    ``raw``, are copied in the new order.
    """
    import anndata as ad

    identity = np.array_equal(order, np.arange(table.n_obs))
    rest = ad.AnnData(
        obs=table.obs,
        var=table.var,
        uns={k: v for k, v in table.uns.items() if k != "spatialdata_attrs"},
        obsm=dict(table.obsm),
        varm=dict(table.varm),
        obsp=dict(table.obsp),
        varp=dict(table.varp),
        layers={k: v for k, v in table.layers.items() if k is not None},  # X is `layers[None]` (anndata.py:659)
    )
    rest = rest.copy() if identity else rest[order].copy()
    X = table.X
    if hasattr(X, "dask"):
        X = _compute_in_order(X, order)
    elif X is not None and not (identity and (isinstance(X, np.ndarray) or sp.issparse(X))):
        X = X[order]
    rest.X = X
    if table.raw is not None:  # anndata writes raw too (`iter_outer`, anndata/utils.py:442-457)
        raw = table.raw.to_adata()
        rest.raw = raw if identity else raw[order].copy()
    return rest


def _compute_in_order(X: Any, order: np.ndarray) -> Any:
    """A 2-d dask array with its rows in `order`, computed one row block at a time into the result.

    Only the result and one block are held, where ``X[order].compute()`` holds the blocks and their concatenation. A
    sparse X is computed twice: once for the length of each row, which fixes where each row goes, then for the values.
    """
    blocks = X.rechunk({1: -1}).to_delayed().ravel()
    starts = np.cumsum((0, *X.chunks[0][:-1]))
    dest = np.empty(X.shape[0], np.int64)
    dest[order] = np.arange(X.shape[0])  # source row -> row of the result
    if not sp.issparse(X._meta):
        out = np.empty(X.shape, X.dtype)
        for start, block in zip(starts, blocks, strict=True):
            b = np.asarray(block.compute())
            out[dest[start : start + len(b)]] = b
        return out
    lengths = np.zeros(X.shape[0], np.int64)
    for start, block in zip(starts, blocks, strict=True):
        b = sp.csr_matrix(block.compute())
        lengths[dest[start : start + b.shape[0]]] = np.diff(b.indptr)
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    data = np.empty(indptr[-1], X.dtype)
    indices = np.empty(indptr[-1], np.int32 if X.shape[1] < np.iinfo(np.int32).max else np.int64)
    for start, block in zip(starts, blocks, strict=True):
        b = sp.csr_matrix(block.compute())
        row = np.repeat(np.arange(b.shape[0]), np.diff(b.indptr))
        at = indptr[dest[start + row]] + np.arange(b.nnz) - b.indptr[row]
        data[at], indices[at] = b.data, b.indices
    return sp.csr_matrix((data, indices, indptr), shape=X.shape)


def write_collection_dataset(
    group: zarr.Group,
    *,
    key: str,
    table: AnnData,
    images: Mapping[str, ArrayLike],
    labels: Mapping[str, ArrayLike] = MappingProxyType({}),
    block_px: int = 1024,
    n_obs_per_chunk: int = 64,
    shard_size: int | str = "1GB",
    image_chunk: int = 256,
    image_shard: int = 4096,
    compressors: Iterable[BytesBytesCodec] = (ZstdCodec(level=3),),
    attrs: Mapping[str, Any] = MappingProxyType({}),
) -> None:
    """Write a table and its pixel arrays as a dataset of a spatialbatch collection.

    The cells are located by ``table.obsm["spatial"]``, their centres ``(x, y)`` in pixel indices of the arrays (the
    centre of pixel ``j`` is at ``j``), which all share one pixel grid. Tiles are not fixed here: :class:`spatialbatch.Loader` cuts a ``crop_px`` square around
    each centre at read time. What it writes at ``group[key]``:

    - the table, with its rows sorted along a Morton (Z-order) curve of the centres, written with
      :func:`annbatch.write_sharded`;
    - ``images/<name>`` and ``labels/<name>``: the arrays as sharded zarr v3 arrays, copied one shard at a time.

    Parameters
    ----------
    group
        The collection group, must be zarr v3, opened without consolidated metadata (zarr refuses to edit a store that
        has it). It is marked as an annbatch collection.
    key
        The name of the dataset in `group`, ``dataset_<i>`` for it to be found by :class:`SpatialDatasetCollection`.
    table
        The rows of the dataset, with the centres in ``obsm["spatial"]``.
    images
        Images ``(c, y, x)`` by name, as numpy, zarr or dask arrays. Dask arrays are computed block by block.
    labels
        Labels ``(y, x)`` by name, on the grid of the images.
    block_px
        Side in pixels of the grid used to order cells. Match it to the window the loader reads (1024 by default).
    n_obs_per_chunk, shard_size
        Chunking of the table, see :func:`annbatch.write_sharded`.
    image_chunk, image_shard
        Side in pixels of the chunks and of the shards of the images and labels. `image_shard` must be divisible by
        `image_chunk`.
    compressors
        Compressors for the images and labels. The table uses the :func:`annbatch.write_sharded` default.
    attrs
        Further entries of the dataset's ``spatialbatch`` attribute, such as the source of the data.
    """
    if "spatial" not in table.obsm:
        raise ValueError("`table.obsm['spatial']` must hold the centre (x, y) of every cell in pixels.")
    centres = np.asarray(table.obsm["spatial"], dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError(f"`table.obsm['spatial']` must be of shape (n_obs, 2), got {centres.shape}.")
    if not images and not labels:
        raise ValueError("At least one of `images` and `labels` must be given.")
    for kind, arrays, ndim in (("images", images, 3), ("labels", labels, 2)):
        for name, array in arrays.items():
            if array.ndim != ndim:
                raise ValueError(f"{kind} must have {ndim} dimensions, but `{name}` has shape {array.shape}.")
    grids = {tuple(a.shape[-2:]) for a in (*images.values(), *labels.values())}
    if len(grids) != 1:
        raise ValueError(f"All images and labels must share one pixel grid, got the shapes (y, x) {sorted(grids)}.")
    encoding = {
        "encoding-version": CENTRES_VERSION,
        "centres": "obsm/spatial",
        "images": list(images),
        "labels": list(labels),
        "block_px": block_px,
    }
    if clash := encoding.keys() & attrs.keys():
        raise ValueError(f"`attrs` must not set {sorted(clash)}, which are written by spatialbatch.")
    if not np.isfinite(centres).all():
        raise ValueError("`table.obsm['spatial']` must hold finite centres.")
    order = morton_order(centres[:, 0], centres[:, 1], block_px)
    adata = _in_order(table, order)
    _write_dataset(
        group,
        adata,
        key=key,
        images=images,
        labels=labels,
        encoding={**encoding, **attrs},
        n_obs_per_chunk=n_obs_per_chunk,
        shard_size=shard_size,
        image_chunk=image_chunk,
        image_shard=image_shard,
        compressors=compressors,
    )


def write_sharded_sdata(
    group: zarr.Group,
    sdata: SpatialData,
    *,
    key: str,
    region: str = "cell_circles",
    images: Sequence[str] = ("morphology_focus",),
    labels: Sequence[str] = (),
    table_name: str = "table",
    cs: str = "global",
    tile_dim_in_units: float | None = None,
    tile_scale: float = 1.0,
    block_px: int = 1024,
    n_obs_per_chunk: int = 64,
    shard_size: int | str = "1GB",
    image_chunk: int = 256,
    image_shard: int = 4096,
    compressors: Iterable[BytesBytesCodec] = (ZstdCodec(level=3),),
) -> None:
    """Write one backed :class:`spatialdata.SpatialData` as a dataset of a spatialbatch collection.

    The counterpart of :func:`annbatch.write_sharded`. What it writes at ``group[key]``:

    - the table, with its rows sorted along a Morton (Z-order) curve of the tile centers so that consecutive rows
      are spatially close, written with :func:`annbatch.write_sharded`, plus ``obsm["tile_origin"]``, the first
      pixel ``(x, y)`` of each tile;
    - ``images/<name>`` and ``labels/<name>``: the full-resolution arrays as sharded zarr v3 arrays, chunked so that
      a tile touches few chunks and sharded so that a request is a few large reads.

    Only elements whose pixel grid is the coordinate system `cs` (no rotation or scaling) can be written, because a
    tile is then a plain slice. Elements such as a rotated H&E image are tiled in place instead, see
    :meth:`spatialbatch.Loader.add_sdata`.

    The tiles are fixed at write time (encoding 0.1.0). :func:`write_collection_dataset` writes datasets that are
    tiled at read time instead.

    Parameters
    ----------
    group
        The destination group, must be zarr v3.
    sdata
        A SpatialData read from zarr, whose table annotates only `region`.
    key
        The name of the dataset in `group`, ``dataset_<i>`` for it to be found by :class:`SpatialDatasetCollection`.
    region, table_name, cs, tile_dim_in_units, tile_scale
        See :meth:`spatialbatch.Loader.add_sdata` and :class:`spatialbatch.Loader`.
    images, labels
        Names of the elements to write.
    block_px
        Side in pixels of the grid used to order cells. Match it to the window the loader reads (1024 by default).
    n_obs_per_chunk, shard_size
        Chunking of the table, see :func:`annbatch.write_sharded`.
    image_chunk, image_shard
        Side in pixels of the chunks and of the shards of the images and labels. `image_shard` must be divisible by
        `image_chunk`.
    compressors
        Compressors for the images and labels. The table uses the :func:`annbatch.write_sharded` default.
    """
    if not sdata.is_backed():
        raise ValueError("`sdata` must be backed by a zarr store (use `spatialdata.read_zarr`).")
    validate_inputs(sdata, region, images, labels, cs, table_name)
    for name in (*images, *labels):
        if not is_identity_xy(element_to_cs_matrix(sdata[name], cs)):
            raise NotImplementedError(
                f"`{name}` is not in the pixel grid of `{cs}`, so its tiles are not slices and it cannot be written to a "
                "collection. Tile it in place with `Loader.add_sdata` instead."
            )
    table, tiles = aligned_tiles(
        sdata, region, table_name, cs, tile_scale=tile_scale, tile_dim_in_units=tile_dim_in_units
    )
    geometry = tile_starts(tiles)
    order = morton_order(tiles["x"].to_numpy(), tiles["y"].to_numpy(), block_px)

    adata = _in_order(table, order)
    adata.obsm["tile_origin"] = np.stack([geometry.x0, geometry.y0], axis=1)[order].astype(np.int32)
    path = str(sdata.path)
    _write_dataset(
        group,
        adata,
        key=key,
        images={name: scale0(path, "images", name) for name in images},
        labels={name: scale0(path, "labels", name) for name in labels},
        encoding={
            "encoding-version": "0.1.0",
            "tile_hw": [geometry.height, geometry.width],
            "cs": cs,
            "region": region,
            "images": list(images),
            "labels": list(labels),
            "block_px": block_px,
            "source": path,
        },
        n_obs_per_chunk=n_obs_per_chunk,
        shard_size=shard_size,
        image_chunk=image_chunk,
        image_shard=image_shard,
        compressors=compressors,
    )


def open_dataset_group(
    group: zarr.Group,
    images: Sequence[str],
    labels: Sequence[str],
    annotations: Sequence[str] | None,
    crop_px: int | None = None,
) -> tuple[Any, pd.DataFrame | None, pd.DataFrame, _Item]:
    """Open a dataset of a collection for :class:`spatialbatch.Loader`.

    A dataset written by :func:`write_collection_dataset` (encoding 0.2.0) is tiled ``crop_px`` square around
    ``obsm["spatial"]``. One written by :func:`write_sharded_sdata` (encoding 0.1.0) keeps the tiles it was written
    with, and `crop_px` is ignored.
    """
    from .loader import _Item, _Source

    meta = group.attrs.get(ENCODING_KEY)
    if meta is None:
        raise ValueError(f"{group.name} was not written by `write_collection_dataset` or `write_sharded_sdata`.")
    missing = [n for n in images if n not in meta["images"]] + [n for n in labels if n not in meta["labels"]]
    if missing:
        raise ValueError(
            f"{missing} are not in the collection, which has images {meta['images']} and labels {meta['labels']}."
        )
    X, obs, var, obsm = _read_table(group, annotations)
    coords = np.asarray(obsm["spatial"][...]) if "spatial" in obsm else None
    if _version(meta["encoding-version"]) >= _version(CENTRES_VERSION):
        if crop_px is None:
            raise ValueError(f"{group.name} is tiled at read time: pass `crop_px` to the Loader.")
        if coords is None:
            raise ValueError(f"{group.name} is tiled at read time but its table has no `obsm['spatial']`.")
        start = round_half_up(coords - crop_px / 2)
        geometry = TileGeometry(start[:, 0], start[:, 1], crop_px, crop_px)
        # the writer's grid cells (write_collection_dataset sorts by these codes), for the loader's block sampler
        blocks = morton_codes(coords[:, 0], coords[:, 1], meta["block_px"])
    else:
        if crop_px is not None:
            warnings.warn(
                f"{group.name} was written with fixed tiles of {meta['tile_hw']} pixels, so `crop_px` is ignored.",
                UserWarning,
                stacklevel=4,
            )
        origin = np.asarray(obsm["tile_origin"][...]).astype(np.int64)
        height, width = meta["tile_hw"]
        geometry = TileGeometry(origin[:, 0], origin[:, 1], height, width)
        blocks = None  # sorted by tile centres in the source's coordinates, which are not stored
    sources: dict[str, _Source] = {}
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            array = group[kind][name]
            sources[name] = _Source(kind, array, array.shape[0] if kind == "images" else None, array.dtype)
    return X, obs, var, _Item(geometry, sources, coords, blocks)


def _read_table(
    group: zarr.Group, annotations: Sequence[str] | None
) -> tuple[Any, pd.DataFrame | None, pd.DataFrame, Mapping[str, Any]]:
    """``X`` (backed), the ``obs`` columns the loader yields, the var index and the backed ``obsm`` of a dataset.

    What :func:`annbatch.utils.load_all_aligned` reads, except that only the `annotations` columns of ``obs`` are read
    (all of them for ``None``, none for an empty list): ``obs`` is the largest part of a dataset held in memory
    (the cell ids and every column, for every cell), and the loader keeps it for the whole run.
    """
    import pandas as pd
    from annbatch.utils import _read_backed
    from anndata.io import read_elem

    var = group["var"]
    var_frame = pd.DataFrame(index=pd.Index(read_elem(var[var.attrs.get("_index")])))
    obsm = {
        k: _read_backed(v)
        for k, v in (group["obsm"].members() if "obsm" in group else [])
        if v.attrs.get("encoding-type") in {"csr_matrix", "array"}
    }
    obs_group = group["obs"]
    if annotations is None:
        obs = read_elem(obs_group)
    elif len(annotations) == 0:
        obs = None
    else:
        columns = list(obs_group.attrs.get("column-order", []))
        if missing := [c for c in annotations if c not in columns]:
            raise KeyError(f"{group.name}: obs has no columns {missing}; it has {columns}")
        index = pd.Index(read_elem(obs_group[obs_group.attrs["_index"]]))
        obs = pd.DataFrame({c: read_elem(obs_group[c]) for c in annotations}, index=index)
    if obs is not None and len(obs.columns) == 0:
        obs = None
    return _read_backed(group["X"]), obs, var_frame, obsm


def _contract_version(group: zarr.Group) -> str | None:
    """The version in ``uns["sddb_contract"]`` of a dataset, or ``None`` if it has no contract."""
    from anndata.io import read_elem

    try:
        element = group["uns"]["sddb_contract"]["version"]
    except KeyError:
        return None
    return str(read_elem(element))


def _check_contract(group: zarr.Group) -> bool:
    """Check a dataset that declares an sddb contract.

    With :mod:`sddb` installed, the dataset is validated by ``sddb.contract.validate_dataset``. Without it, only the
    version string is checked.

    Returns
    -------
    Whether the dataset was validated in full: ``False`` if it declares a contract that could not be validated.

    Raises
    ------
    ValueError
        If the version does not parse or the validator reports issues.
    """
    version = _contract_version(group)
    if version is None:
        return True
    try:
        from sddb.contract import validate_dataset
    except ImportError:
        if not _SEMVER.match(version):
            raise ValueError(
                f"{group.name} declares the sddb contract version {version!r}, which is not semver."
            ) from None
        return False
    if issues := validate_dataset(group):
        raise ValueError(f"{group.name} does not conform to the sddb contract {version}:\n- " + "\n- ".join(issues))
    return True


def _cache_dir() -> Path:
    """Where validated collections are recorded: ``$SPATIALBATCH_CACHE_DIR``, else ``~/.cache/spatialbatch``."""
    return Path(os.environ.get("SPATIALBATCH_CACHE_DIR") or Path.home() / ".cache" / "spatialbatch")


def _validation_key(groups: Sequence[zarr.Group], root: zarr.Group) -> str | None:
    """What a validation of the collection depends on, hashed; ``None`` if it cannot be fixed cheaply.

    The key covers the root's consolidated metadata (every array's shape, dtype, chunks and attributes, so a
    rewritten or resized element changes it), each dataset's ``uns["sddb_contract"]`` (whose values are array data,
    not metadata) and the version of the validator (``sddb``). Without consolidated metadata there is no key.
    """
    from anndata.io import read_elem

    if root.metadata.consolidated_metadata is None:
        return None
    try:
        import sddb
        import sddb.contract as contract
    except ImportError:
        return None
    parts = [json.dumps(root.metadata.to_dict(), sort_keys=True, default=str)]
    parts += [getattr(sddb, "__version__", "unknown"), getattr(contract, "CONTRACT_VERSION", "unknown")]
    for group in groups:
        try:
            parts.append(json.dumps(read_elem(group["uns"]["sddb_contract"]), sort_keys=True, default=str))
        except KeyError:
            parts.append("")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _check_collection(groups: Sequence[zarr.Group], root: zarr.Group) -> bool:
    """:func:`_check_contract` on every dataset, once per collection: a collection that passed is recorded.

    A pass is recorded under :func:`_validation_key` in this process and as a file in :func:`_cache_dir`, so a later
    open of the same collection, in this process or another, skips the checks. A collection with issues is checked
    again on every open. Returns whether every dataset was validated in full.
    """
    key = _validation_key(groups, root)
    marker = None if key is None else _cache_dir() / "validated" / key
    if key in _VALIDATED or (marker is not None and marker.is_file()):
        _VALIDATED.add(key)
        return True
    validated = [_check_contract(group) for group in groups]  # every dataset, not only up to the first
    if all(validated) and marker is not None:
        _VALIDATED.add(key)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"collection": str(root.store), "datasets": [g.name for g in groups]}))
        except OSError as e:  # a read-only home: the check runs again next time
            warnings.warn(f"could not record the validation in {marker}: {e}", UserWarning, stacklevel=3)
    return all(validated)


#: Validation keys of the collections that passed in this process.
_VALIDATED: set[str | None] = set()


class SpatialDatasetCollection(DatasetCollection):
    """A collection of spatially sorted, sharded SpatialData datasets, for :meth:`spatialbatch.Loader.use_collection`.

    This is :class:`annbatch.DatasetCollection` with a different writer. Iteration over the datasets,
    :attr:`is_empty` and :meth:`obs` are inherited. A dataset is one SpatialData, written by
    :func:`write_sharded_sdata` or :func:`write_collection_dataset`: unlike annbatch there is no mixing of rows across files on disk, since pixels are not
    rows that can be moved around. Slides are mixed when sampling, as the chunks drawn by one request come from any
    dataset.

    Parameters
    ----------
    group
        The base location of the collection, a :class:`zarr.Group` or a path ending in ``.zarr``.
    mode
        The mode the store is opened with if `group` is a path.

    Examples
    --------
    >>> collection = SpatialDatasetCollection("collection.zarr")
    >>> collection.add_sdatas(["a.zarr", "b.zarr"], images=["morphology_focus"], tile_dim_in_units=224)
    >>> loader = Loader(images=["morphology_focus"], batch_size=32, shuffle=True).use_collection(collection)
    """

    def __init__(self, group: zarr.Group | str | PathLike[str], *, mode: str = "a"):
        super().__init__(group, mode=mode)  # type: ignore[arg-type]
        if not isinstance(self._group, zarr.Group):
            raise TypeError("A SpatialDatasetCollection is always zarr.")
        self._cached_var: pd.DataFrame | None = None

    @property
    def var(self) -> pd.DataFrame:
        """The var shared by the datasets: the var index and the columns that are equal in every dataset.

        Every dataset of a collection has one var index (the loader needs the same columns of `X`); this checks it.
        A column that differs between datasets, such as an sddb collection's per-dataset ``var["present"]``, is left
        out. In an sddb collection the index holds the Ensembl ids and the columns ``gene_ids``, ``feature_types`` and
        ``original_var_name`` (the source name, e.g. the gene symbol).

        Raises
        ------
        ValueError
            If the collection is empty or its datasets have different var indexes.
        """
        from anndata.io import read_elem

        if self._cached_var is not None:
            return self._cached_var
        groups = list(self)
        if not groups:
            raise ValueError("SpatialDatasetCollection is empty")
        frames = [read_elem(g["var"]) for g in groups]
        first = frames[0]
        for group, frame in zip(groups[1:], frames[1:], strict=True):
            if not frame.index.equals(first.index):
                raise ValueError(f"{group.name} has another var index than {groups[0].name}")
        shared = [c for c in first.columns if all(c in f.columns and f[c].equals(first[c]) for f in frames[1:])]
        self._cached_var = first[shared].copy()
        return self._cached_var

    @property
    def n_obs_per_dataset(self) -> list[int]:
        """The rows (cells) of each dataset, in the order the loader reads them. Reads only the ``obs`` index shape."""
        out = []
        for group in self:
            obs = group["obs"]
            out.append(_elem_len(obs[obs.attrs["_index"]]))
        return out

    @property
    def n_obs(self) -> int:
        """The rows of the collection: the ``n_obs`` of a :class:`~spatialbatch.Loader` that uses it."""
        return sum(self.n_obs_per_dataset)

    @property
    def offsets(self) -> np.ndarray:
        """The loader row of each dataset's first row: dataset ``d`` holds rows ``offsets[d]`` to ``offsets[d + 1]``.

        Its length is the number of datasets plus one, and the last value is :attr:`n_obs`. A batch's ``index`` (the
        loader row, with ``return_index=True``) is ``offsets[batch["dataset"]]`` plus the row within its dataset; see
        :meth:`locate`.
        """
        return np.concatenate([[0], np.cumsum(self.n_obs_per_dataset)]).astype(np.int64)

    def locate(self, index: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(dataset, row within the dataset)`` of each loader row in `index`, e.g. a batch's ``index``.

        The row within the dataset is the row of that dataset's ``obs`` and ``X``, which are sorted along the
        Morton curve when written, so not the source table's order.
        """
        index = np.asarray(index, dtype=np.int64)
        offsets = self.offsets
        if index.size and (index.min() < 0 or index.max() >= offsets[-1]):
            raise IndexError(f"index must be in [0, {offsets[-1]})")
        dataset = np.searchsorted(offsets, index, side="right") - 1
        return dataset, index - offsets[dataset]

    def add_adatas(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Not supported: a dataset of this collection holds pixels, written by :meth:`add_sdatas`."""
        raise TypeError("A SpatialDatasetCollection holds pixels too: use `add_sdatas`.")

    def add_sdatas(self, sdata_paths: Iterable[str | PathLike[str]], **kwargs: Any) -> None:
        """Write SpatialData stores to the collection, each as a new dataset.

        Parameters
        ----------
        sdata_paths
            Paths to SpatialData zarr stores.
        **kwargs
            Passed to :func:`write_sharded_sdata`.
        """
        import spatialdata as sd

        self._cached_var = None
        paths = list(sdata_paths)
        first = len(self._dataset_keys)
        for i, path in enumerate(tqdm(paths, desc="Adding to collection")):
            write_sharded_sdata(self._group, sd.read_zarr(path), key=f"{DATASET_PREFIX}_{first + i}", **kwargs)


def _elem_len(elem: Any) -> int:
    """The length of an anndata column from its metadata: an array, or a group encoding one.

    anndata writes a string index as the group ``nullable-string-array`` (``values`` and ``mask``) where nullable
    strings are on, and a categorical as ``codes`` and ``categories``.
    """
    if isinstance(elem, zarr.Array):
        return int(elem.shape[0])
    for key in ("values", "codes"):
        if key in elem:
            return int(elem[key].shape[0])
    raise ValueError(f"cannot tell the length of {elem.name} (encoding {elem.attrs.get('encoding-type')!r})")


def spatial_split(collection: SpatialDatasetCollection, frac: float) -> tuple[list[slice], list[slice]]:
    """``(train, held_out)``: the last `frac` of every dataset's rows held out, as loader ``rows``.

    Within each dataset the rows are sorted along a Morton curve of the cell positions, so a dataset's last rows are a
    compact region of its slide (the last quadrants of the curve), not a random sample. Each list holds one slice per
    dataset and can be passed as ``Loader(rows=...)`` or :meth:`~spatialbatch.Loader.set_rows`.

    Parameters
    ----------
    collection
        The collection the loader uses.
    frac
        The share of each dataset's rows to hold out, in ``(0, 1)``. A dataset holds out ``round(frac * n)`` rows.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    offsets = collection.offsets
    train, held_out = [], []
    for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        cut = stop - round(frac * (stop - start))
        train.append(slice(start, cut))
        held_out.append(slice(cut, stop))
    return train, held_out
