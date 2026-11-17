from __future__ import annotations

import numpy as np
import pytest
import zarr

from spatialbatch import Loader, SpatialDatasetCollection, write_collection_dataset
from spatialbatch.samplers import BlockSampler
from spatialbatch.samplers import _block_sampler as block_module
from spatialbatch.utils import morton_codes

# -- the sampler on its own ----------------------------------------------------------------------------------------


def _starts(n_obs: int, seed: int = 0) -> np.ndarray:
    sizes = np.random.default_rng(seed).integers(1, 40, n_obs)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    return starts[starts < n_obs]


def _requests(sampler: BlockSampler, n_obs: int, monkeypatch=None, worker=None) -> list[dict]:
    if monkeypatch is not None:
        monkeypatch.setattr(block_module, "get_torch_worker_info", lambda: worker)
    return list(sampler.sample(n_obs))


def _check_stream(requests: list[dict], starts: np.ndarray, selected: np.ndarray) -> None:
    """Every selected row once; each slice inside one block; each block one run of the stream, from its first row."""
    slices = [s for r in requests for s in r["requests"]]
    rows = np.concatenate([np.arange(s.start, s.stop) for s in slices])
    np.testing.assert_array_equal(np.sort(rows), selected)
    first = np.searchsorted(starts, [s.start for s in slices], side="right") - 1
    last = np.searchsorted(starts, [s.stop - 1 for s in slices], side="right") - 1
    np.testing.assert_array_equal(first, last)
    runs = first[np.concatenate([[True], first[1:] != first[:-1]])]
    assert len(np.unique(runs)) == len(runs)  # a block is never split over two places of the stream
    block_of_selected = np.searchsorted(starts, selected, side="right") - 1
    block_first = {b: selected[np.flatnonzero(block_of_selected == b)[0]] for b in np.unique(block_of_selected)}
    run_heads = np.flatnonzero(np.concatenate([[True], first[1:] != first[:-1]]))
    assert all(slices[i].start == block_first[first[i]] for i in run_heads)


def test_requests_are_shuffled_whole_blocks_and_cover_every_row_once():
    starts = _starts(1000)
    sampler = BlockSampler(16, 4, 16, rng=np.random.default_rng(0))
    sampler.block_starts = starts
    requests = _requests(sampler, 1000)
    sizes = [sum(s.stop - s.start for s in r["requests"]) for r in requests]
    assert sizes[:-1] == [64] * (len(sizes) - 1) and sizes[-1] == 1000 - 64 * (len(sizes) - 1)
    _check_stream(requests, starts, np.arange(1000))
    heads = [r["requests"][0].start for r in requests]
    assert heads != sorted(heads)  # blocks in random order
    for r, n in zip(requests, sizes, strict=True):  # rows shuffled in memory, split into batches
        indices = np.concatenate(r["splits"])
        np.testing.assert_array_equal(np.sort(indices), np.arange(n))
        assert not np.array_equal(indices, np.arange(n))
    assert sum(len(r["splits"]) for r in requests) == sampler.n_batches(1000) == -(-1000 // 16)


def test_rows_cut_blocks_to_the_selected_rows():
    starts = _starts(1000, seed=1)
    mask = np.random.default_rng(2).random(1000) < 0.6
    mask[300:450] = False
    sampler = BlockSampler(16, 4, 16, rows=mask, rng=np.random.default_rng(0))
    sampler.block_starts = starts
    _check_stream(_requests(sampler, 1000), starts, np.flatnonzero(mask))


def test_workers_take_disjoint_shares_of_the_blocks(monkeypatch):
    from annbatch.samplers._utils import WorkerInfo

    starts = _starts(1000, seed=3)
    seen = []
    for worker in range(3):
        sampler = BlockSampler(16, 4, 16, drop_last=True, rng=np.random.default_rng(5))
        sampler.block_starts = starts
        requests = _requests(sampler, 1000, monkeypatch, WorkerInfo(worker, 3))
        rows = np.concatenate([np.arange(s.start, s.stop) for r in requests for s in r["requests"]])
        used = np.concatenate([np.concatenate(r["splits"]) for r in requests])
        assert used.size % 16 == 0 and rows.size - used.size < 16  # drop_last: each worker drops < batch_size rows
        seen.append(rows)
    every = np.concatenate(seen)  # a worker's last request is not yielded if it holds less than a batch
    assert len(np.unique(every)) == every.size > 1000 - 3 * 16  # disjoint, and together all but < 3 batches
    blocks = [set(np.searchsorted(starts, rows, side="right") - 1) for rows in seen]
    assert not (blocks[0] & blocks[1]) and not (blocks[0] & blocks[2]) and not (blocks[1] & blocks[2])


def test_needs_block_starts():
    with pytest.raises(RuntimeError, match="block starts"):
        list(BlockSampler(16, 4, 16).sample(100))


# -- through the Loader, on a collection of several datasets ----------------------------------------------------------

SHAPE = (300, 340)
N_DATASETS, N_CELLS, BLOCK_PX = 4, 200, 64


@pytest.fixture
def collection(tmp_path):
    import anndata as ad
    import pandas as pd
    from scipy import sparse as sp

    group = zarr.open_group(tmp_path / "c.zarr", mode="a")
    for i in range(N_DATASETS):
        rng = np.random.default_rng(i)
        image = rng.integers(1, 255, (1, *SHAPE), dtype=np.uint8)
        table = ad.AnnData(
            sp.random(N_CELLS, 5, density=0.3, format="csr", dtype=np.float32, random_state=i),
            obs=pd.DataFrame(index=[f"c{i}-{j}" for j in range(N_CELLS)]),
        )
        table.obsm["spatial"] = rng.uniform([0, 0], [SHAPE[1], SHAPE[0]], (N_CELLS, 2))
        write_collection_dataset(
            group,
            key=f"dataset_{i}",
            table=table,
            images={"img": image},
            block_px=BLOCK_PX,
            image_chunk=32,
            image_shard=128,
            n_obs_per_chunk=16,
            shard_size=64,
        )
    return SpatialDatasetCollection(tmp_path / "c.zarr")


def _loader(collection, **kwargs) -> Loader:
    kw = {"images": ["img"], "crop_px": 16, "return_index": True, "batch_size": 20, "shuffle": True}
    return Loader(**{**kw, **kwargs}).use_collection(collection)


def test_block_starts_are_the_writers_grid_cells(collection):
    loader = _loader(collection, chunk_size=20, preload_nchunks=3, chunks="blocks")
    expected = []
    for i, group in enumerate(collection):
        xy = np.asarray(group["obsm"]["spatial"][...])
        codes = morton_codes(xy[:, 0], xy[:, 1], BLOCK_PX)
        assert np.all(np.diff(codes.astype(np.int64)) >= 0)  # the rows are sorted by these codes
        expected.append(i * N_CELLS + np.flatnonzero(np.concatenate([[True], codes[1:] != codes[:-1]])))
    np.testing.assert_array_equal(loader.block_starts, np.concatenate(expected))
    assert 5 < N_DATASETS * N_CELLS / loader.block_starts.size < 10  # ~30 cells of 64 px on 300 x 340 px


def test_loader_blocks_visit_every_row_once_with_the_right_tiles(collection, ref_tile):
    loader = _loader(collection, chunk_size=20, preload_nchunks=3, chunks="blocks", rng=np.random.default_rng(0))
    images = [np.asarray(g["images"]["img"][...]) for g in collection]
    coords = [np.asarray(g["obsm"]["spatial"][...]) for g in collection]
    index = []
    for batch in loader:
        for row, (idx, ds) in enumerate(zip(batch["index"], batch["dataset"], strict=True)):
            cx, cy = coords[ds][idx - ds * N_CELLS]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(images[ds], cx, cy, 16))
        index.append(batch["index"])
    np.testing.assert_array_equal(np.sort(np.concatenate(index)), np.arange(N_DATASETS * N_CELLS))
    assert isinstance(loader.batch_sampler, BlockSampler)


def test_blocks_keep_several_datasets_per_batch(collection):
    def datasets_per_batch(**kwargs) -> float:
        loader = _loader(collection, rng=np.random.default_rng(0), **kwargs)
        return float(np.mean([len(np.unique(b["dataset"])) for b in loader]))

    blocks = datasets_per_batch(chunk_size=20, preload_nchunks=3, chunks="blocks")
    one_chunk = datasets_per_batch(chunk_size=60, preload_nchunks=1)
    assert blocks > 2.5 and one_chunk < 2


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_blocks_with_rows_and_torch_workers(collection, num_workers):
    pytest.importorskip("torch")
    rows = [slice(10, 170), slice(250, 700)]
    loader = _loader(collection, chunk_size=20, preload_nchunks=3, chunks="blocks", rows=rows)
    index = np.concatenate([b["index"].numpy() for b in loader.to_torch(num_workers=num_workers, pin_memory=False)])
    expected = np.r_[10:170, 250:700]
    assert len(np.unique(index)) == index.size and np.isin(index, expected).all()
    if num_workers == 0:
        assert index.size == expected.size
    else:  # drop_last for the workers: each drops at most batch_size - 1 rows
        assert index.size % 20 == 0 and index.size >= expected.size - num_workers * 19


def test_chunks_argument_errors(collection):
    with pytest.raises(ValueError, match="shuffle=True"):
        Loader(images=["img"], crop_px=16, chunks="blocks")
    with pytest.raises(ValueError, match="'rows' or 'blocks'"):
        Loader(images=["img"], crop_px=16, shuffle=True, chunks="cells")
    with pytest.raises(ValueError, match="BlockSampler"):
        Loader(images=["img"], crop_px=16, batch_sampler=BlockSampler(), chunks="blocks")
