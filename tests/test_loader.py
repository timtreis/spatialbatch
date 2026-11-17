from __future__ import annotations

import warnings

import numpy as np
import pytest
from annbatch.samplers import SequentialSampler

from spatialbatch import Loader
from spatialbatch.abc import Sampler
from spatialbatch.samplers import RandomSampler

TILE = 32
KW = {"region": "cells"}
TILES = {"tile_dim_in_units": TILE, "rasterize": True, "rasterize_kwargs": {"target_unit_to_pixels": 1.0}}


def _identity(batch):
    return batch


def _loader(**kwargs) -> Loader:
    return Loader(images=["img"], labels=["lab"], return_index=True, **{**TILES, **kwargs})


def test_tiles_are_the_pixels_under_each_cell(synthetic, ref_tile):
    sdata, raw = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2).add_sdata(sdata, **KW)
    seen = 0
    for batch in loader:
        for row, idx in enumerate(batch["index"]):
            cx, cy = raw["centers"][idx]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raw["image"], cx, cy))
            np.testing.assert_array_equal(batch["labels"]["lab"][row], ref_tile(raw["labels"], cx, cy))
            np.testing.assert_array_equal(batch["coords"][row], (raw["centers"][idx] / 2).astype(np.float32))
            assert batch["obs"]["label"].iloc[row] == idx + 1
            seen += 1
    assert seen == 200


def test_every_cell_is_visited_once_per_epoch(synthetic, make, tmp_path):
    sdata, _ = synthetic
    other, _ = make(tmp_path / "b.zarr", n_cells=150, seed=1)
    loader = _loader(
        batch_size=16, chunk_size=8, preload_nchunks=4, shuffle=True, rng=np.random.default_rng(0)
    ).add_sdatas([sdata, other], **KW)
    assert loader.n_obs == 350
    index = np.concatenate([b["index"] for b in loader])
    np.testing.assert_array_equal(np.sort(index), np.arange(350))
    assert not np.array_equal(index, np.arange(350))  # actually shuffled
    assert sum(len(b["index"]) for b in loader) == 350
    assert len(loader) == -(-350 // 16)


def test_rows_of_each_dataset_get_their_own_pixels(synthetic, make, ref_tile, tmp_path):
    sdata, raw_a = synthetic
    other, raw_b = make(tmp_path / "b.zarr", n_cells=150, seed=1, shape=(280, 300))
    loader = _loader(
        batch_size=25, chunk_size=25, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(2)
    ).add_sdatas([sdata, other], **KW)
    for batch in loader:
        for row, idx in enumerate(batch["index"]):
            raw, local = (raw_a, idx) if idx < 200 else (raw_b, idx - 200)
            cx, cy = raw["centers"][local]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raw["image"], cx, cy))
            assert batch["obs"]["label"].iloc[row] == local + 1


def test_sampler_is_an_annbatch_sampler():
    sampler = RandomSampler(chunk_size=4, preload_nchunks=2, batch_size=4, rng=np.random.default_rng(0))
    assert isinstance(sampler, Sampler)
    flat = np.concatenate([np.arange(s.start, s.stop) for r in sampler.sample(40) for s in r["requests"]])
    np.testing.assert_array_equal(np.sort(flat), np.arange(40))


def test_custom_sampler_and_transform(synthetic):
    sdata, _ = synthetic
    sampler = SequentialSampler(chunk_size=10, preload_nchunks=2, batch_size=10, mask=slice(0, 40))
    loader = Loader(
        images=["img"], batch_sampler=sampler, transform=lambda b: b["images"]["img"].sum(), **TILES
    ).add_sdata(sdata, **KW)
    assert len(list(loader)) == 4


def test_torch_tensors(synthetic):
    torch = pytest.importorskip("torch")
    sdata, _ = synthetic
    batch = next(iter(_loader(batch_size=8, chunk_size=8, preload_nchunks=1, to="torch").add_sdata(sdata, **KW)))
    assert isinstance(batch["images"]["img"], torch.Tensor)
    assert tuple(batch["images"]["img"].shape) == (8, 1, TILE, TILE)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_dataloader(synthetic, num_workers):
    torch_data = pytest.importorskip("torch.utils.data")
    sdata, _ = synthetic
    loader = _loader(
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        shuffle=True,
        drop_last=True,
        rng=np.random.default_rng(0),
        to="torch",
    ).add_sdata(sdata, **KW)
    dataloader = torch_data.DataLoader(loader, batch_size=None, num_workers=num_workers, collate_fn=_identity)
    index = np.concatenate([np.asarray(b["index"]) for b in dataloader])
    assert len(np.unique(index)) == len(index) == 200


def test_errors(synthetic, make, tmp_path):
    sdata, _ = synthetic
    with pytest.raises(ValueError, match="Cannot specify"):
        Loader(batch_sampler=RandomSampler(4, 2, 4), batch_size=2)
    with pytest.raises(ValueError, match="At least one"):
        Loader(images=[])
    with pytest.raises(TypeError, match="add_sdata"):
        Loader().add_adata(None)
    with pytest.raises(ValueError, match="same integer size"):
        Loader(images=["img"], tile_scale=2.0).add_sdata(
            make(tmp_path / "v.zarr", radius=np.linspace(2, 6, 200))[0], **KW
        )
    with pytest.raises(ValueError, match="not valid for the region"):
        _loader().add_sdata(sdata, cs="nope", **KW)
    loader = _loader().add_sdata(sdata, **KW)
    three_channels, _ = make(tmp_path / "c.zarr", seed=3, channels=3)
    with pytest.raises(ValueError, match="same modalities"):
        loader.add_sdata(three_channels, **KW)
    assert loader.n_obs == 200  # the failed add left the loader untouched


def test_concatenated_indptr_holds_offsets_past_the_part_dtype(synthetic):
    """A batch over two datasets where the second adds < 65536 nonzeros after an offset > 65535 (annbatch 0.3.1)."""
    from collections import OrderedDict

    from annbatch import Loader as AnnbatchLoader
    from annbatch.utils import CSRContainer

    sdata, _ = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2).add_sdata(sdata, **KW)
    n_var = loader.n_var

    def part(nnz_per_row: list[int]) -> CSRContainer:
        indptr = np.concatenate([[0], np.cumsum(nnz_per_row)])
        nnz = int(indptr[-1])
        indptr = indptr.astype(np.min_scalar_type(nnz))  # as annbatch's _allocate_out sizes it
        return CSRContainer(
            elems=(np.ones(nnz, np.float32), np.arange(nnz, dtype=np.int32) % n_var, indptr),
            shape=(len(nnz_per_row), n_var),
            dtype=np.float32,
        )

    def outs():
        return OrderedDict([(0, part([40_000, 30_000])), (1, part([3, 4]))])

    assert outs()[1].elems[2].dtype == np.uint8
    with pytest.raises(OverflowError):  # plain annbatch
        AnnbatchLoader._concatenate_outs(loader, outs())
    out = loader._concatenate_outs(outs())
    np.testing.assert_array_equal(out.elems[2], [0, 40_000, 70_000, 70_003, 70_007])
    assert out.elems[0].size == 70_007


# -- rows= and the dataset id ------------------------------------------------------------------------------------------


def test_rows_two_ranges_over_two_datasets_visit_each_row_once(synthetic, make, tmp_path):
    sdata, _ = synthetic
    other, _ = make(tmp_path / "b.zarr", n_cells=150, seed=1)
    rows = [slice(20, 90), slice(230, 300)]  # one range in each dataset (200 + 150 rows)
    loader = _loader(
        batch_size=16, chunk_size=8, preload_nchunks=4, shuffle=True, rng=np.random.default_rng(0), rows=rows
    ).add_sdatas([sdata, other], **KW)
    batches = list(loader)
    index = np.concatenate([b["index"] for b in batches])
    expected = np.r_[20:90, 230:300]
    np.testing.assert_array_equal(np.sort(index), expected)  # each selected row exactly once
    assert not np.array_equal(index, expected)  # shuffled
    assert len(loader) == len(batches) == -(-expected.size // 16)
    dataset = np.concatenate([b["dataset"] for b in batches])
    np.testing.assert_array_equal(dataset, (index >= 200).astype(int))


def test_rows_mask_index_array_and_slices_select_the_same_rows(synthetic, make, tmp_path):
    sdata, _ = synthetic
    other, _ = make(tmp_path / "b.zarr", n_cells=150, seed=1)
    mask = np.zeros(350, bool)
    mask[[*range(5, 40), *range(190, 260), 349]] = True
    seen = []
    for rows in (mask, np.flatnonzero(mask)[::-1], [slice(5, 40), slice(190, 260), slice(349, 350)]):
        loader = _loader(batch_size=8, chunk_size=8, preload_nchunks=2, rows=rows).add_sdatas([sdata, other], **KW)
        seen.append(np.concatenate([b["index"] for b in loader]))
    np.testing.assert_array_equal(seen[0], np.flatnonzero(mask))  # sequential: in row order
    np.testing.assert_array_equal(seen[1], seen[0])
    np.testing.assert_array_equal(seen[2], seen[0])


def test_rows_chunks_keep_contiguous_runs_as_slices():
    from spatialbatch.samplers._rows import _chunks

    assert _chunks(np.arange(10, 30), 8) == [[slice(10, 18)], [slice(18, 26)], [slice(26, 30)]]
    assert _chunks(np.r_[0:5, 9:12], 4) == [[slice(0, 4)], [slice(4, 5), slice(9, 12)]]


def test_rows_rejects_bad_input(synthetic):
    sdata, _ = synthetic
    with pytest.raises(ValueError, match="length n_obs"):
        list(_loader(batch_size=10, chunk_size=10, preload_nchunks=2, rows=np.ones(5, bool)).add_sdata(sdata, **KW))
    with pytest.raises(ValueError, match="no row"):
        list(_loader(batch_size=10, chunk_size=10, preload_nchunks=2, rows=[]).add_sdata(sdata, **KW))


def test_dataset_id_of_each_row(synthetic, make, tmp_path):
    sdata, _ = synthetic
    other, _ = make(tmp_path / "b.zarr", n_cells=150, seed=1)
    loader = _loader(
        batch_size=25, chunk_size=25, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(1)
    ).add_sdatas([sdata, other], **KW)
    for batch in loader:
        np.testing.assert_array_equal(batch["dataset"], (batch["index"] >= 200).astype(int))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_rows_with_torch_workers(synthetic, num_workers):
    torch_data = pytest.importorskip("torch.utils.data")
    sdata, _ = synthetic
    loader = _loader(
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        shuffle=True,
        drop_last=True,
        rng=np.random.default_rng(0),
        rows=[slice(0, 60), slice(100, 180)],
        to="torch",
    ).add_sdata(sdata, **KW)
    dataloader = torch_data.DataLoader(loader, batch_size=None, num_workers=num_workers, collate_fn=_identity)
    index = np.concatenate([np.asarray(b["index"]) for b in dataloader])
    np.testing.assert_array_equal(np.sort(index), np.r_[0:60, 100:180])


# -- to_torch ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
def test_to_torch_sets_up_the_dataloader(synthetic, num_workers):
    torch = pytest.importorskip("torch")
    sdata, _ = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(0))
    loader = loader.add_sdata(sdata, **KW)
    dataloader = loader.to_torch(num_workers=num_workers, pin_memory=False)
    assert dataloader.batch_size is None
    assert dataloader.persistent_workers is (num_workers > 0)
    batches = list(dataloader)
    index = np.concatenate([b["index"].numpy() for b in batches])
    assert all(isinstance(b["images"]["img"], torch.Tensor) for b in batches)
    assert all(isinstance(b["dataset"], torch.Tensor) for b in batches)
    assert len(np.unique(index)) == len(index)
    if num_workers > 1:  # drop_last set for the workers: each worker drops at most its last batch_size - 1 rows
        assert loader.batch_sampler._drop_last and len(index) % 10 == 0 and len(index) >= 200 - num_workers * 9
    else:
        assert len(index) == 200
    assert len(list(dataloader)) == len(batches)  # persistent workers serve a second epoch


def test_to_torch_warns_under_the_file_descriptor_strategy(synthetic):
    torch = pytest.importorskip("torch")
    sdata, _ = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2, shuffle=True).add_sdata(sdata, **KW)
    before = torch.multiprocessing.get_sharing_strategy()
    try:
        torch.multiprocessing.set_sharing_strategy("file_descriptor")
        with pytest.warns(UserWarning, match="file_descriptor"):
            loader.to_torch(num_workers=2, pin_memory=False)
        torch.multiprocessing.set_sharing_strategy("file_system")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader.to_torch(num_workers=2, pin_memory=False)
        assert not [w for w in caught if "file_descriptor" in str(w.message)]
    finally:
        torch.multiprocessing.set_sharing_strategy(before)


def test_to_torch_refuses_what_annbatch_refuses(synthetic):
    pytest.importorskip("torch")
    sdata, _ = synthetic
    with pytest.raises(ValueError, match="does not shuffle"):
        _loader(batch_size=10, chunk_size=10).add_sdata(sdata, **KW).to_torch(num_workers=2)
    explicit = _loader(batch_size=10, chunk_size=10, shuffle=True, drop_last=False).add_sdata(sdata, **KW)
    with pytest.raises(ValueError, match="drop_last=True"):
        explicit.to_torch(num_workers=2)
    with pytest.raises(ValueError, match="batch_size"):
        explicit.to_torch(batch_size=4)


def test_set_rows_without_workers(synthetic):
    sdata, _ = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2, rows=slice(0, 50)).add_sdata(sdata, **KW)
    assert np.concatenate([b["index"] for b in loader]).tolist() == list(range(50))
    loader.set_rows(slice(150, 200))
    assert np.concatenate([b["index"] for b in loader]).tolist() == list(range(150, 200))
    loader.set_rows(None)
    assert len(np.concatenate([b["index"] for b in loader])) == 200
    with pytest.raises(ValueError, match="no row"):
        loader.set_rows([])


def test_set_rows_reaches_persistent_workers(synthetic):
    pytest.importorskip("torch")
    sdata, _ = synthetic
    loader = _loader(
        batch_size=10, chunk_size=10, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(0), rows=slice(0, 120)
    ).add_sdata(sdata, **KW)
    dataloader = loader.to_torch(num_workers=2, pin_memory=False)
    train = np.concatenate([b["index"].numpy() for b in dataloader])
    pids = {w.pid for w in dataloader._iterator._workers}
    loader.set_rows(slice(120, 200))
    val = np.concatenate([b["index"].numpy() for b in dataloader])
    assert {w.pid for w in dataloader._iterator._workers} == pids  # the same worker processes
    assert set(train) <= set(range(120)) and len(train) >= 120 - 2 * 9
    assert set(val) <= set(range(120, 200)) and len(val) >= 80 - 2 * 9
    loader.set_rows(slice(0, 120))
    assert set(np.concatenate([b["index"].numpy() for b in dataloader])) <= set(range(120))


@pytest.mark.parametrize("num_workers", [0, 1, 3])
def test_evaluate_covers_every_row_once_with_workers(synthetic, num_workers):
    pytest.importorskip("torch")
    sdata, _ = synthetic
    # chunks of 8 rows: with 3 workers a share of 80 or 200 rows is 4, 3, 3 or 9, 8, 8 chunks, which 16 does not
    # divide, so a shuffled pass with workers would drop rows (annbatch needs 16 to divide a request of 2 chunks)
    loader = _loader(
        batch_size=16, chunk_size=8, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(0), rows=slice(0, 120)
    ).add_sdata(sdata, **KW)
    dataloader = loader.to_torch(num_workers=num_workers, pin_memory=False)
    train = np.concatenate([b["index"].numpy() for b in dataloader])
    assert set(train) <= set(range(120))
    for rows, expected in ((slice(120, 200), np.arange(120, 200)), (None, np.arange(200))):
        loader.set_rows(rows, evaluate=True)
        batches = [b["index"].numpy() for b in dataloader]
        index = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(index), expected)  # every row, none twice
        assert all(0 < len(b) <= 16 for b in batches)
        if num_workers == 0:
            np.testing.assert_array_equal(index, expected)  # in row order
            assert len(batches) == len(loader)
    loader.set_rows(slice(0, 120))  # back to training: shuffled, and with workers the incomplete batches dropped
    train = np.concatenate([b["index"].numpy() for b in dataloader])
    assert set(train) <= set(range(120))
    if num_workers > 1:
        assert len(train) % 16 == 0


def test_evaluation_loader_without_shuffle_takes_workers(synthetic):
    pytest.importorskip("torch")
    sdata, _ = synthetic
    loader = _loader(batch_size=16, chunk_size=8, preload_nchunks=2).add_sdata(sdata, **KW)
    with pytest.raises(ValueError, match="does not shuffle"):
        loader.to_torch(num_workers=2)
    loader.set_rows(None, evaluate=True)
    index = np.concatenate([b["index"].numpy() for b in loader.to_torch(num_workers=2, pin_memory=False)])
    np.testing.assert_array_equal(np.sort(index), np.arange(200))


@pytest.mark.parametrize("fused", [False, True])
def test_windows_group_tiles_by_centre_and_read_them_whole(tmp_path, monkeypatch, ref_tile, fused):
    import zarr
    from zarr.codecs import ZstdCodec
    from zarr.core.sync import sync

    import spatialbatch.loader as sl
    from spatialbatch.utils import TileGeometry

    image = np.random.default_rng(0).integers(1, 255, (2, 200, 200), dtype=np.uint8)
    array = zarr.create_array(
        tmp_path / "a.zarr",
        shape=image.shape,
        dtype=image.dtype,
        chunks=(2, 16, 16),
        shards=(2, 64, 64),
        compressors=ZstdCodec(level=3),
    )
    array[...] = image
    if fused:
        array = sl._fused(array)
    monkeypatch.setattr(sl, "_WINDOW_PX", 64)
    # Centres (x, y) of 20 px tiles. Rows 0 and 1 have their centres in window (1, 1); row 0's tile starts in
    # window (0, 0), where row 2 lies, and crosses the grid line at 64. Row 3 crosses the array border.
    centres = np.array([[70, 70], [100, 90], [60, 60], [195, 5]])
    geometry = TileGeometry(centres[:, 0] - 10, centres[:, 1] - 10, 20, 20)
    windows = []
    getitem = zarr.AsyncArray.getitem

    async def spy(self, selection, *args, **kwargs):
        windows.append(tuple((s.start, s.stop) for s in selection[-2:]))
        return await getitem(self, selection, *args, **kwargs)

    monkeypatch.setattr(zarr.AsyncArray, "getitem", spy)
    out = np.empty((4, 2, 20, 20), np.uint8)
    source = sl._Source("images", array, 2, array.dtype)
    sync(Loader(images=["img"], crop_px=20)._fetch_tiles(source, geometry, np.arange(4), out))
    assert sorted(windows) == [((0, 15), (185, 200)), ((50, 70), (50, 70)), ((60, 100), (60, 110))]
    for row, (cx, cy) in enumerate(centres):
        np.testing.assert_array_equal(out[row], ref_tile(image, cx, cy, 20))
