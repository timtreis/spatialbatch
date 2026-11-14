# spatialbatch

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/rushin/spatialbatch/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/spatialbatch/badge/

A minibatch loader for on-disk [SpatialData][spatialdata] stores: tiles of every image and labels element, centered on each cell,
together with the cell's expression and annotations, streamed in large preloaded chunks.

`spatialbatch` is built on, and has [annbatch][] as a dependency. It follows the same design: a `Sampler` emits requests for chunks of
consecutive cells, one request is read from disk at once (concurrently for every modality, through async `zarr` and
optionally [zarrs-python][]), shuffled in memory and split into batches. No torch `DataLoader` is needed.
It reimplements `spatialstream.dataloaders.XeniumDataset`, which it reproduces exactly (see `tests/test_parity.py`).

```python
import spatialdata as sd
import zarr
from spatialbatch import Loader

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})  # optional, see docs/zarr-configuration.md

loader = Loader(
    images=["morphology_focus"],
    labels=["cell_labels"],
    tile_dim_in_units=224,
    rasterize=True,
    rasterize_kwargs={"target_unit_to_pixels": 1.0},
    return_annotations=["cell_labels"],
    batch_size=32,
    shuffle=True,
    to="torch",
).add_sdata(sd.read_zarr("xenium.zarr"))

for batch in loader:
    batch["images"]["morphology_focus"]  # (32, 1, 224, 224)
    batch["labels"]["cell_labels"]  # (32, 224, 224)
    batch["X"]  # (32, n_genes) sparse CSR expression
```

For the best throughput, write the stores once into a spatially sorted, sharded `SpatialDatasetCollection`:

```python
from spatialbatch import Loader, SpatialDatasetCollection

collection = SpatialDatasetCollection("collection.zarr")
collection.add_sdatas(["a.zarr", "b.zarr"], images=["morphology_focus"], labels=["cell_labels"], tile_dim_in_units=224)
loader = Loader(images=["morphology_focus"], batch_size=32, shuffle=True, tile_dim_in_units=224).use_collection(collection)
```

### Try it

A 400 µm crop of an Atera H&E slide (2,326 cells), written as a collection by spatialdata-db (about 15 MB):

```bash
pip install "spatialbatch @ git+https://github.com/timtreis/spatialbatch@feat/obsm-centres"
curl -O https://exampledata.scverse.org/spatialdata-db/fixtures/sddb-contract/0.1.0/atera-breast-crop.zarr.zip && unzip -q atera-breast-crop.zarr.zip
python -c 'from spatialbatch import Loader, SpatialDatasetCollection as C
loader = Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=32).use_collection(C("sddb-contract-0.1.0-atera-breast-crop.zarr"))
b = next(iter(loader)); print(b["images"]["he"].shape, b["labels"]["cells"].shape, b["X"].shape)'
```

### Train/validation split

`rows=` restricts sampling to a subset of the collection's global rows: a slice, a list of slices, an integer array
or a boolean mask. Global rows run through the datasets in order (dataset `i` starts at the sum of the earlier
datasets' `n_obs`), and `batch["dataset"]` says which dataset each row came from. Within a dataset the rows are
Morton-sorted on `obsm["spatial"]`, so a range of rows is a compact region of the slide, and a contiguous range is
read as contiguous chunks.

```python
import numpy as np
from spatialbatch import Loader, SpatialDatasetCollection

collection = SpatialDatasetCollection("atera-he.zarr")
n = [g["obsm"]["spatial"].shape[0] for g in collection]  # rows per dataset, in collection order
start = np.concatenate([[0], np.cumsum(n)])

held_out = 5  # one whole slide for validation
val = [slice(start[held_out], start[held_out + 1])]
# plus the last 10% of the rows of every other slide: spatial tails, compact regions
val += [slice(start[i + 1] - n[i] // 10, start[i + 1]) for i in range(len(n)) if i != held_out]
is_val = np.zeros(start[-1], bool)
for s in val:
    is_val[s] = True

common = dict(images=["he"], labels=["cells"], crop_px=224, batch_size=32, return_index=True)
train = Loader(**common, shuffle=True, rows=~is_val).use_collection(collection)
valid = Loader(**common, rows=val).use_collection(collection)
```

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install spatialbatch:

<!--
1. Add the latest release of `spatialbatch` from [PyPI][] to your `uv` project:

   ```bash
   uv add spatialbatch
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install spatialbatch
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/rushin/spatialbatch.git  # (or `uv add`)
   ```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[spatialdata]: https://spatialdata.scverse.org
[annbatch]: https://github.com/scverse/annbatch
[zarrs-python]: https://zarrs-python.readthedocs.io
[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/rushin/spatialbatch/issues
[tests]: https://github.com/rushin/spatialbatch/actions/workflows/test.yaml
[documentation]: https://spatialbatch.readthedocs.io
[changelog]: https://spatialbatch.readthedocs.io/page/changelog.html
[api documentation]: https://spatialbatch.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/spatialbatch
[venv]: https://docs.python.org/3/tutorial/venv.html
