# CLAUDE.md

This file provides guidance to coding agents when working in this repository.

## Overview

SynapseNet (`synapse_net`) is a Python package for deep-learning segmentation and analysis of synapses in electron microscopy, mainly electron tomography. It ships U-Net models for segmenting synaptic vesicles, active zones, compartments, mitochondria, cristae, and ribbon-synapse structures, plus analysis tools (distance measurement, morphometry, vesicle pooling) and IMOD import/export. It is exposed three ways: a **napari plugin**, a **CLI** (console scripts), and a **Python library**.

## Environment & Install

The dependencies for this package are available via conda. We are in the process of removing dependency on nifty and vigra in favor of bioimage-cpp.
This will enable pure pip installation.

Here is how to set-up an environment with a clean installation.

```bash
conda env create -f environment.yaml   # creates env "synapse-net"
conda activate synapse-net
pip install -e .                        # or: pip install --no-deps -e .  (as CI does)
```

IMOD export commands additionally require the external IMOD suite installed.

## Testing

```bash
python -m unittest discover -s test -v                       # full suite (what CI runs)
python -m unittest test.test_inference.TestInference         # one test class
python -m unittest test.test_inference.TestInference.test_run_segmentation  # one test
```

Tests are **network-dependent**: they download pretrained models (via `pooch` from GWDG ownCloud) and sample data the first time they run, into the OS cache dir (`pooch.os_cache("synapse-net")`). A first run is slow and requires internet; offline runs will fail on model/sample download, not on a code bug.

## Common commands

```bash
# Build API docs (pdoc, google docstring format) — output to tmp/
python build_doc.py -o

# CLI entry points (defined in setup.py console_scripts):
synapse_net.run_segmentation -i <mrc-or-dir> -o <out-dir> -m vesicles_3d
synapse_net.export_to_imod_points  -i <mrc-dir> -s <seg-dir> -o <mod-dir>
synapse_net.export_to_imod_objects -i <mrc-dir> -s <seg-dir> -o <mod-dir>
synapse_net.run_supervised_training -h
synapse_net.run_domain_adaptation   -h
synapse_net.visualize_vesicle_pools -h
```

## Architecture

The package is organized by capability. The big picture spans several modules:

### Models & the model registry (`synapse_net/inference/inference.py`)
Models are identified by a **`model_type` string** (`vesicles_3d`, `vesicles_2d`, `vesicles_cryo`, `active_zone`, `compartments`, `mitochondria`/`mitochondria2`, `cristae`/`cristae2`/`cristae3`, `ribbon`, plus CLI-only `vesicles_*` variants). `_get_model_registry()` maps each name to a sha256 + GWDG ownCloud download URL and fetches via `pooch`. `get_model(model_type)` downloads (if needed) and `torch.load`s the checkpoint.

**Voxel-size scaling is central.** Each model was trained at a specific resolution (`get_model_training_resolution`, in nm/axis). `compute_scale_from_voxel_size(voxel_size, model_type)` produces a zyx scale factor so input data is resized to match training resolution before inference and resized back after. Always pair a model with the right `scale` rather than feeding raw-resolution data.

### Inference dispatch
`run_segmentation(image, model, model_type, tiling, scale, **kwargs)` is the top-level entry. It branches on `model_type` to a per-structure function in its own module: `inference/vesicles.py`, `active_zone.py`, `compartments.py`, `mitochondria.py`, `cristae.py`, `ribbon_synapse.py`, `actin.py`. The `ribbon` path is special — it runs multi-output prediction then optional post-processing that requires a vesicle segmentation passed via `extra_segmentation`.

`inference/util.py` holds the shared machinery:
- `get_prediction` / `get_prediction_torch_em` — tiled prediction via `torch_em.util.prediction.predict_with_halo`. **Tiling is a dict** `{"tile": {"z","y","x"}, "halo": {"z","y","x"}}`; note `get_prediction` subtracts `2*halo` from each tile dim before calling the torch_em layer.
- `_Scaler` — applies/reverts the voxel-size scale (order-0 for segmentations).
- `inference_helper` — drives file-in/file-out batch segmentation over a folder; used by the CLI.
- A `bioimageio` prediction path exists in `get_prediction` but currently raises `NotImplementedError` (model format migration in progress; see the `bioimage-cpp-migration` branch).

Per-structure **post-processing** lives in `inference/postprocessing/` (vesicles, compartments, membranes, ribbon, presynaptic_density), kept separate from raw prediction.

### Training (`synapse_net/training/`)
Built on `torch_em` U-Nets (`UNet2d`, `AnisotropicUNet`). Two regimes:
- `supervised_training.py` — needs data + manual labels. Can init weights from a pretrained `model_type`.
- `domain_adaptation.py` — unsupervised student-teacher (mean-teacher) adaptation to a new condition without labels; only works if the source model already partially detects the structure.
- `semisupervised_training.py`, `transform.py` — supporting pieces.

### Tools (`synapse_net/tools/`)
- `cli.py` — argparse wrappers behind the console scripts.
- Napari widgets (`segmentation_widget.py`, `distance_measure_widget.py`, `morphology_widget.py`, `vesicle_pool_widget.py`, `postprocessing_widget.py`, `size_filter_widget.py`), all subclassing `base_widget.py` and registered in `synapse_net/napari.yaml`. **When adding/renaming a widget, command, reader, or sample, update `napari.yaml` and the `setup.py` entry points to match.**
- `volume_reader.py` — napari reader for `.mrc`/`.rec`/`.h5`.

### Analysis & I/O
- `distance_measurements.py`, `morphology.py` — quantification of segmentation results.
- `imod/to_imod.py`, `imod/export.py` — convert segmentations to IMOD point models (spheres per vesicle) or closed-contour object models.
- `file_utils.py` — I/O. `read_mrc` reads `.mrc`/`.rec`; **voxel sizes are converted Angstrom→nm by dividing by 10**, and data axes are flipped to match Python order. Also reads OME-Zarr and streams from the CryoET Data Portal (`cryoet_data_portal`, `s3fs` — both optional imports).
- `ground_truth/` — annotation/GT-curation utilities (vesicle matching, shape refinement, edge filtering).

### `scripts/`
Project- and collaboration-specific scripts (subfolders `cooper`, `inner_ear`, `cryo`, `rizzoli`, `wichmann`, `portal`, `baselines`, …) tied to the publication and ongoing collaborations. These are **not** part of the installable package and not covered by tests; many paths are gitignored. Don't treat them as library API.

## Conventions

- Spatial dicts use explicit axis keys (`{"x":..,"y":..,"z":..}`); arrays are zyx. Be careful that tiling/scale lists are in zyx order while voxel-size dicts are keyed by axis name.
- Public functions use Google-style docstrings (docs are generated with `pdoc -d google`); match that style when adding API.
- Optional heavy deps (`zarr`, `s3fs`, `cryoet_data_portal`) are imported defensively and guarded — keep new optional integrations the same way.
