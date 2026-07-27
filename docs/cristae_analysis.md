# Cristae Analysis — what the widget does

Napari widget: `synapse_net/tools/cristae_analysis_widget.py` (`CristaeAnalysisWidget`).
Analysis backend: `synapse_net/cristae_analysis.py`.

It takes a **crista segmentation** (binary) and a **mitochondria instance segmentation**, and
produces per‑mitochondrion morphometrics plus napari result layers. Everything is quantified in
physical units (nm / nm² / nm³) using the voxel size.

---

## Inputs & settings (widget)

- **Crista Mask** (Labels layer) — binary crista segmentation.
- **Mito Segmentation** (Labels layer) — instance‑labelled mitochondria (0 = background).
- **Advanced settings:**
  - `voxel_size` (nm; `0` = auto‑detect from the layer metadata).
  - `mm_thickness` — membrane shell thickness (nm, default 8).
  - `border_gap` — distance from volume faces where membrane is suppressed (nm; `0` = use `mm_thickness`).
  - `show_membranes` (**Show Membrane Mesh**) — also add the eroded‑mito (lumen) inner surface — the
    surface the junction geodesics run along — as a **mesh** (napari Surface layer).
  - `save_path` — optional CSV/XLSX export of the results table.

> **Units caveat:** MRC/H5 files often store voxel size in **Ångström**; the correct value in nm
> is that ÷ 10 (e.g. 8.681 Å → 0.8681 nm). Wrong voxel size changes the membrane band width and
> therefore the junction/contact results.

---

## Pipeline (widget `on_run`)

1. Read the crista + mito arrays and resolve the voxel size (metadata or the spin‑box).
2. Read the source layer's `scale`/`translate` so result layers overlay the raw data correctly.
3. **Approximate the mitochondrial membrane** (`approximate_membrane`).
4. **Compute per‑mitochondrion statistics** (`compute_mito_crista_statistics`) → a pandas DataFrame.
5. Optionally add the **Membrane Mesh** surface layer (eroded‑mito lumen surface) if **Show Membrane Mesh** is enabled.
6. Add the **Crista‑Membrane Junctions** labels layer (each junction a unique ID; `translucent_no_depth` blending), or notify if none.
7. Attach the stats table to the mito layer and optionally save it (CSV/XLSX).

---

## Command-line interface

The same analysis is available headlessly via the `synapse_net.run_cristae_analysis` console script
(`cristae_analysis_cli` in `synapse_net/tools/cli.py`). It takes a crista segmentation and a
mitochondria instance segmentation, computes `compute_mito_crista_statistics`, and writes one
`<stem>_cristae_analysis.csv` per input pair. It accepts a single file **or** a directory for each
input (batch); for a directory the input subfolder structure is mirrored under the output, and
already-present tables are skipped unless `--force` is given.

Segmentations are read as tif when no key is given, or from an hdf5 dataset when a key is given — so
the crista and mito segmentations can live in separate files or in the *same* hdf5 file under
different keys.

| Flag | Default | Meaning |
|---|---|---|
| `--crista_path` / `-c` | required | Crista segmentation — file or directory. |
| `--mito_path` / `-m` | required | Mitochondria instance segmentation — file or directory. |
| `--output_path` / `-o` | required | Output directory for the result table(s). |
| `--crista_key` | — | HDF5 dataset key for the crista segmentation (omit → tif). |
| `--mito_key` | — | HDF5 dataset key for the mito segmentation (omit → tif). |
| `--voxel_size` | — | Voxel size in **nm**, applied to all inputs. |
| `--tomogram_path` | — | Raw tomogram (mrc/rec), file or directory, used to read the voxel size when `--voxel_size` is omitted. |
| `--membrane_thickness` | `8.0` | Membrane shell thickness (nm). |
| `--border_gap` | thickness | Distance from volume faces where the membrane is suppressed (nm). |
| `--method` | `skip` | Orientation anisotropy mode: `skip` / `fast` / `exact`. |
| `--membrane_mode` | `slice_2d` | Membrane shell construction: `slice_2d` / `shell_3d`. |
| `--n_jobs` | `-1` | Workers for the per-mitochondrion computation (`-1` = all cores). |
| `--force` | off | Over-write already present result tables. |
| `--verbose` / `-v` | off | Show a progress bar over the mitochondria of each file. |

One of `--voxel_size` or `--tomogram_path` must be given (the physical-unit results are wrong without
the correct voxel size — see the units caveat above).

Examples:

```bash
# Single pair of tif segmentations, explicit voxel size.
synapse_net.run_cristae_analysis -c crista.tif -m mito.tif -o results/ --voxel_size 0.868 -v

# Both segmentations in one hdf5 file under different keys.
synapse_net.run_cristae_analysis \
    -c seg.h5 --crista_key labels/cristae \
    -m seg.h5 --mito_key   labels/mitochondria \
    -o results/ --voxel_size 0.868

# Directory batch, reading the voxel size from the matching raw tomograms.
synapse_net.run_cristae_analysis \
    -c crista_dir/ -m mito_dir/ --tomogram_path tomo_dir/ -o results/
```

The output CSV has one row per mitochondrion with the columns documented in **Output columns** below.

---

## Computational steps & math

### 1. Membrane approximation — `approximate_membrane`
The membrane is approximated as the **outer boundary shell of the mitochondrion**. For 3D data it is
done **per Z‑slice** with a 2D disk of radius `mm_thickness / voxel_xy`:
`membrane = mito & ~binary_erosion(mito, disk(r))` — i.e. the rim removed by the erosion. Per‑slice
(not a 3D ball) prevents the shell bleeding across slices when the shape changes in Z. A separable
Z‑only erosion (radius `mm_thickness / voxel_z`) then adds **Z‑caps** where a mito column truly ends in
Z; ends clipped by a volume Z‑face stay uncapped (`border_value=1` plus the `border_gap` trim below).
Voxels within `border_gap` of any volume face are removed (so clipped mito edges aren't treated as
membrane).
The eroded interior is the **lumen**, whose inner‑wall surface is the mesh the junction geodesics run
along (see §5 and the **Membrane Mesh** layer). At mesh time the lumen surface is trimmed to the
certain region — the border zone within `border_gap` of the volume faces is cropped off, so it ends
where the membrane ends instead of flaring to full mito width — and **left open** there (not capped),
so geodesics cannot shortcut across a fabricated cap at a clipped face (`_open_trimmed_mesh`).
*Math:* binary morphological erosion. *Perf:* cropped to the mito bbox, empty slices skipped, and
the per‑slice erosion is parallelised.

### 2. Crista orientation — `compute_crista_orientation`
Computes the **structure-tensor eigenvalues** of the binary crista mask via
`bioimage_cpp.filters.structure_tensor_eigenvalues` (a fast C++ routine): `inner_sigma` is the
derivative scale (a minimal 1 voxel), `outer_sigma = neighborhood_size_nm / voxel` is the integration
scale (default 30 nm). The per‑voxel eigenvalues (descending) give
**anisotropy = λ_max / (λ_min + ε)**, and the reported `crista_orientation_anisotropy` is the mean
over crista voxels.
*Interpretation:* high → strongly directional (parallel lamellae), ~1 → isotropic/tubular. It is a
**magnitude, rotation‑invariant** — it says *how* laminar, not *which* direction (the orientation
direction / eigenvectors are not computed).

### 3. Crista→membrane proximity — `compute_crista_proximity`
`distance_transform_edt` of the non‑membrane, sampled by voxel size → distance (nm) from each crista
voxel to the nearest membrane voxel; reports the **median** (`avg_crista_to_membrane_nm`).
*Math:* Euclidean distance transform.

### 4. Crista‑membrane junctions — `detect_contact_sites`
A junction is the **pure intersection** of the crista mask and the membrane band
(`crista & membrane`) — **no hidden dilation/erosion**, so junctions correspond exactly to the
visible overlap of the two layers. Overlaps are grouped into discrete junctions by
**26‑connectivity connected components** (`scipy.ndimage.label`). Reports `contact_voxel_count`,
`crista_junction_count`, `contact_volume_nm3`, and a labelled junction array (unique ID per junction).
> Because the junction reach equals the membrane band width (`mm_thickness`), a too‑large thickness
> at fine voxel sizes flags cristae that merely come near the boundary.

### 5. Junction distances & clustering — `compute_junction_distances`
This measures the **spacing between junctions *along the membrane surface*** — not the straight line
through the lumen — and summarises it with a Clark–Evans nearest‑neighbour index. Distances are
**surface geodesics on the eroded‑mito (lumen) surface**: the single‑wall triangle mesh at the
membrane's inner edge, `_surface_mesh(mito & ~membrane)` via marching cubes (verts in nm, `+1` pad).
Each junction's centroid is snapped to its nearest mesh vertex (`scipy.spatial.cKDTree`, in the
`(centroid+1)*sampling` padded frame so it matches the mesh), and the pairwise geodesic matrix is
computed with `bioimage_cpp.distance.geodesic_distances_mesh` (needs **bioimage‑cpp ≥ 0.6.0**; `0`
diagonal, `+inf`→`NaN` across disconnected components, threaded). From the matrix it derives the
nearest‑neighbour distances (`mean_/median_nn_junction_distance_nm`; `NaN` when a mito has <2 reachable
junctions) and a **Clark‑Evans clustering index** `R = mean_NN / (0.5 · √(A/n))` with `A` = mito
surface area and `n` = junction count (`junction_clustering_index`: <1 clustered, ≈1 random,
>1 dispersed; flat‑surface CSR approximation).
> The bioimage‑cpp geodesic backend (`bioimage_cpp.distance.geodesic_distances_mesh`) is a hard
> dependency — **there is no other backend**. A mito with no usable lumen mesh (empty/degenerate)
> yields `NaN` junction‑distance columns.

**"Missing" pairs.** Two junctions on **different connected components** of the lumen surface have no
along‑surface path → that pair is `NaN` and is ignored by the nearest‑neighbour / clustering
summaries. `membrane_mode="slice_2d"` (the default) caps true (non‑clipped) Z‑ends but its XY shell
can still fragment across slices more readily than `"shell_3d"`. If junctions you expect to be
connected are not, switch to `"shell_3d"`, increase `mm_thickness`, or check the mito segmentation.

**What shifts with the membrane settings.** The membrane‑dependent outputs — `crista_junction_count`,
the contact counts, `avg_crista_to_membrane_nm`, and the junction distances — change with
`membrane_mode`, `mm_thickness` and `border_gap`. Volumes, surface areas, thickness and orientation
do **not**.

### 6. Crista morphology & surfaces — `compute_crista_morphology`, `_surface_area`
Surface area via **marching cubes** (`skimage.measure.marching_cubes` + `mesh_surface_area`) on the
mask (padded so edge‑touching objects are closed). Average crista thickness = **2 × mean EDT value at
the medial axis** (medial axis = local maxima of the distance transform, `skimage.morphology.local_maxima`).
The same `_surface_area` gives the **mito outer‑membrane surface**, and
`crista_to_mito_surface_ratio = crista surface / mito surface` (a size‑normalised "crista surface
density"; can exceed 1 for folded cristae).

### Per‑mito assembly — `compute_mito_crista_statistics`
Iterates mitochondria (`skimage.measure.regionprops` for labels + bounding boxes), crops each to its
bbox, and runs the steps above. Rows are ordered by label; results are independent of the parallelism
settings.

### Code map

| Step | Function | File |
|---|---|---|
| Membrane shell (`slice_2d` / `shell_3d`) | `approximate_membrane` | `synapse_net/cristae_analysis.py` |
| Crista orientation anisotropy | `compute_crista_orientation` (`bioimage_cpp.filters.structure_tensor_eigenvalues`; `_downsampled_orientation_anisotropy` for `fast`) | `synapse_net/cristae_analysis.py` |
| Crista→membrane proximity | `compute_crista_proximity` | `synapse_net/cristae_analysis.py` |
| Junctions (crista ∩ membrane) | `detect_contact_sites` | `synapse_net/cristae_analysis.py` |
| Junction geodesic distances | `compute_junction_distances` → `_junction_matrix_mesh` → `bioimage_cpp.distance.geodesic_distances_mesh` | `synapse_net/cristae_analysis.py` |
| Eroded‑mito (lumen) surface mesh | `_open_trimmed_mesh` → `_surface_mesh` (eroded‑mito lumen, trimmed to the certain region and left open at clipped volume faces) | `synapse_net/cristae_analysis.py` |
| Surface area & thickness | `compute_crista_morphology`, `_surface_area`, `_medial_axis_thickness_nm` | `synapse_net/cristae_analysis.py` |
| Per‑mito assembly | `_single_mito_row` / `compute_mito_crista_statistics` | `synapse_net/cristae_analysis.py` |
| Widget | `CristaeAnalysisWidget` | `synapse_net/tools/cristae_analysis_widget.py` |

---

## Output columns (one row per mitochondrion)

Each row is one mitochondrion instance. Every physical quantity uses the voxel size, so it is only
correct if the voxel size is right (nm, not Å — see the units caveat above).

| Column | Units | How it's computed | Notes |
|---|---|---|---|
| `mito_label_id` | — | The instance label of this mitochondrion in the mito segmentation (`regionprops`). | Identifies the row; not a measurement. |
| `mito_touches_border` | bool | `True` if the mito's bounding box lies within the border‑gap radius of any volume face. | Border‑touching mitos are clipped by the field of view, so their volumes/areas are underestimates. |
| `mito_volume_nm3` | nm³ | (number of mito voxels) × voxel volume. | — |
| `crista_volume_nm3` | nm³ | (number of crista voxels inside this mito) × voxel volume. | — |
| `crista_fraction` | — (0–1) | `crista_volume_nm3 / mito_volume_nm3`. | Fraction of the mito volume occupied by cristae. |
| `contact_voxel_count` | count | Number of voxels in `crista & membrane` (the literal overlap). | Scales with the membrane band width (`mm_thickness`) and voxel size. |
| `crista_junction_count` | count | Number of 26‑connected components of the contact voxels. | One "junction" = one connected crista–membrane contact patch. |
| `contact_volume_nm3` | nm³ | `contact_voxel_count` × voxel volume. | — |
| `avg_crista_to_membrane_nm` | nm | Euclidean distance transform from each crista voxel to the nearest membrane voxel, summarised as the **median**. | Despite the `avg_` name this is the **median**, not the mean. |
| `mean_nn_junction_distance_nm` | nm | For each junction, the nearest‑neighbour **surface geodesic** to another junction (along the eroded‑mito lumen mesh); averaged over junctions. | `NaN` if the mito has <2 junctions reachable on the same connected surface. See §5. |
| `median_nn_junction_distance_nm` | nm | As above, but the median of the per‑junction nearest‑neighbour geodesics. | Same `NaN` condition as the mean. |
| `junction_clustering_index` | — | Clark–Evans `R = mean_NN / (0.5·√(A/n))`, with `A` = mito surface area, `n` = junction count. | `<1` clustered, `≈1` random, `>1` dispersed. Flat‑surface CSR approximation. `NaN` without a valid mean_NN / surface area. |
| `crista_orientation_anisotropy` | — | Mean over crista voxels of `λ_max/(λ_min+ε)` from the crista structure tensor. | Depends on the **orientation mode**: `skip` (default) = `NaN`; `fast` = 2× downsampled (relative indicator only, **not** comparable in magnitude to `exact`); `exact` = full resolution. Rotation‑invariant (magnitude, not direction). |
| `cristae_surface_area_nm2` | nm² | Marching‑cubes surface area of the crista mask. | — |
| `mito_surface_area_nm2` | nm² | Marching‑cubes surface area of the mito (outer‑membrane surface). | Also used as `A` in the clustering index. |
| `crista_to_mito_surface_ratio` | — | `cristae_surface_area_nm2 / mito_surface_area_nm2`. | Size‑normalised "crista surface density"; **can exceed 1** for folded cristae. |
| `avg_thickness_nm` | nm | `2 × mean(EDT)` at the crista medial axis (medial axis = local maxima of the crista distance transform). | Mean local thickness of the crista sheets. |

Napari layers added: **Crista‑Membrane Junctions** (labels, unique ID per junction, rendered with
`translucent_no_depth` blending), and — when **Show Membrane Mesh** is enabled — a **Membrane Mesh**
surface layer (the eroded‑mito lumen inner surface; see §5). Layers inherit the source layer's
`scale`/`translate`. (The **Preview** button additionally shows the membrane as a **Membrane Mask**
labels layer.)

---

## Volume measurement vs IMOD (voxel count vs contour volume)

`mito_volume_nm3` and `crista_volume_nm3` are **voxel counts**. If you export the *same* masks to
IMOD and read the volume back there, IMOD reports a **contour‑enclosed volume** that is
systematically a little **smaller** — negligibly for mitochondria (~1–2 %), but noticeably for thin
cristae (up to **~10–12 %**), which also shifts `crista_fraction` (e.g. 0.116 by voxel count vs
0.106 from IMOD). **This is a definitional difference, not an error in either tool**, and it is
expected. The two definitions are described below so results are comparable across tools.

### What SynapseNet computes — discrete voxel volume
`_single_mito_row` / `compute_mito_crista_statistics` in `synapse_net/cristae_analysis.py`:

```python
mito_local   = mito_crop == label          # this mitochondrion instance
crista_local = crista_crop & mito_local     # cristae voxels inside this mito
voxel_vol    = np.prod(sampling)            # voxel_z * voxel_y * voxel_x   (nm^3)
mito_vol     = float(mito_local.sum())   * voxel_vol
crista_vol   = float(crista_local.sum()) * voxel_vol
```

Every segmented voxel is counted as **one full rectangular box** (`voxel_x × voxel_y × voxel_z`). No
surface or contour is involved — this is the exact discrete volume of the segmentation mask. It is the
more accurate volume estimate for a voxel segmentation, and it does not under‑measure thin structures.

### What IMOD computes — per‑slice contour ("Cylinder") volume
The SynapseNet exporter `write_segmentation_to_imod` in `synapse_net/imod/to_imod.py` (CLI
`synapse_net.export_to_imod_objects` → `imod_object_cli`) binarizes the mask and runs
**`imodauto -E 1 -u`** to trace, on each Z‑slice, a **closed contour** around the thresholded pixels.
**`imodinfo`** then reports:

> **Cylinder Volume = Σ_slices (contour polygon area) × slice thickness**

The traced contour runs approximately through the **centers of the outer boundary pixels**, so each
per‑slice contour encloses roughly **half a voxel less** than the pixels it bounds — all the way around
its perimeter. IMOD's volume is therefore a *surface/contour‑bounded* volume, not a voxel count.

### Why the gap is much larger for cristae than mitochondria
The fractional loss per slice ≈ **(contour perimeter ÷ 2) / contour area** — it scales with each
contour's **perimeter‑to‑area ratio** and is essentially **independent of the object's Z‑extent**:

- A **mitochondrion** is a compact blob → low perimeter/area → **~1–2 %** smaller in IMOD.
- **Cristae** appear in each slice as **thin curved strips** (1–2 voxels wide) → very high
  perimeter/area → up to **~10–12 %** smaller in IMOD.

Because cristae lose a larger fraction than mitochondria, **`crista_fraction` is not tool‑invariant
either**. Do **not** change SynapseNet to match IMOD — that would inherit IMOD's thin‑structure
under‑measurement. For cross‑tool comparison, either compare voxel counts on both sides or expect the
offset above.

Verified on the test masks (0.8 nm isotropic voxels) and synthetic shapes driven through the real
`imodauto`→`imodinfo` pipeline:

| object | voxel count | IMOD Cylinder Volume | gap |
|---|---|---|---|
| mitochondrion (compact) | 634,555 | 633,402 | **0.18 %** |
| cristae (this test crop) | 77,492 | 76,662 | **1.07 %** |
| synthetic 20‑voxel‑wide strip (mito‑like) | 18,000 | 17,838 | **0.9 %** |
| synthetic 1‑voxel‑thin strip (crista‑like) | 900 | 795 | **11.7 %** |

(The gap on real cristae reaches the ~11 % seen in practice once the cristae are as thin/folded as
they are in full tomograms; the small test crop above is chunkier.)

### Reproduce it

Analyze the masks in SynapseNet (voxel‑count volume):

```bash
synapse_net.run_cristae_analysis -c crista.tif -m mito.tif -o results/ --voxel_size 0.8
# -> results/<stem>_cristae_analysis.csv has mito_volume_nm3 / crista_volume_nm3 (voxel counts)
```

Export the *same* mask to an IMOD `.mod` and read the contour volume back. Either via the CLI
(needs a source `.mrc`/`.rec` for the voxel size, matched to the segmentation `.tif` by filename):

```bash
synapse_net.export_to_imod_objects -i tomo_dir/ -s seg_dir/ -o mod_dir/
imodinfo -F mod_dir/<name>.mod        # read "Cylinder Volume"
```

…or directly in Python (this is exactly what was used to produce the table above):

```python
import re, subprocess, numpy as np, mrcfile, imageio.v3 as iio
from synapse_net.imod.to_imod import write_segmentation_to_imod

voxel_nm = 0.8
mask = (iio.imread("mito.tif") > 0).astype("uint8")

# write_segmentation_to_imod reads the voxel size from an .mrc; IMOD stores it in Angstrom (nm * 10)
with mrcfile.new("ref.mrc", data=np.zeros(mask.shape, "uint8"), overwrite=True) as f:
    f.voxel_size = voxel_nm * 10        # 8.0 Angstrom
    f.update_header_from_data()

write_segmentation_to_imod("ref.mrc", mask, "mito.mod", separate_instances=False)

out = subprocess.run(["imodinfo", "-F", "mito.mod"], capture_output=True, text=True).stdout
cyl = float(re.search(r"Cylinder Volume\s*=\s*([0-9.eE+-]+)", out).group(1))
count = int(mask.sum())
print("SynapseNet voxel volume (nm^3):", count * voxel_nm**3)
print("IMOD contour volume    (nm^3):", cyl * voxel_nm**3)   # Cylinder Volume is in voxel^3 here
print("gap: %.2f %%" % (100 * (count - cyl) / count))
```

> **Units note:** with this export IMOD leaves the model pixel size at 1 (`imodinfo` prints
> `UNITS: pixels`, `PIX SIZE = 1`), so **"Cylinder Volume" is in voxel³** — multiply by `voxel_nm**3`
> to compare to SynapseNet's nm³. The **percentage gap is unit‑independent**, so comparing gaps needs
> no conversion. `separate_instances` does not affect the volume of a single, non‑touching object
> (it only trims the interface between *touching* instances).

**Function / command reference:**

| Role | What | Where |
|---|---|---|
| SynapseNet voxel volume | `_single_mito_row`, `compute_mito_crista_statistics` | `synapse_net/cristae_analysis.py` |
| SynapseNet → IMOD export | `write_segmentation_to_imod` (runs `imodauto -E 1 -u`) | `synapse_net/imod/to_imod.py` |
| Export CLI | `synapse_net.export_to_imod_objects` → `imod_object_cli` | `synapse_net/tools/cli.py` |
| IMOD contour tracing | `imodauto` | IMOD suite |
| IMOD volume readout | `imodinfo -F` → "Cylinder Volume" | IMOD suite |

---

## Libraries used
- **NumPy** — arrays and numerics.
- **SciPy** — `scipy.ndimage` (`binary_erosion`, `distance_transform_edt`, `label`,
  `center_of_mass`); `scipy.spatial.cKDTree` (snap junction centroids to mesh vertices).
- **bioimage‑cpp** (≥ 0.6.0, required) — `bioimage_cpp.distance.geodesic_distances_mesh` for junction
  surface geodesics, and `bioimage_cpp.filters.structure_tensor_eigenvalues` for the crista
  orientation anisotropy.
- **scikit‑image** — `measure.marching_cubes`, `measure.mesh_surface_area`, `measure.regionprops`;
  `morphology.disk`, `morphology.local_maxima`.
- **pandas** — results table. **tqdm** — progress. **napari** / **qtpy** — the widget/UI.
- **concurrent.futures** (`ThreadPoolExecutor`) + **psutil** — parallelism and memory‑aware worker caps.

---

## Performance & memory (important facts)
- **Adaptive parallelism** across mitochondria: with many mitos it parallelises **across** them on a
  `concurrent.futures.ThreadPoolExecutor` — the heavy per‑mito stages (structure tensor, EDT,
  geodesics) are GIL‑releasing C++, so threads scale them — with each worker's inner stages kept
  single‑threaded (the EDT/geodesic solvers are called with `number_of_threads=1`); with few/one
  dominant mito it runs them serially and parallelises **within** the mito (junction geodesic
  threads). Exactly one level of parallelism is active — no oversubscription.
- **Memory‑aware caps** (`_available_memory_bytes`/`_bounded_workers` via `psutil`) bound the worker/
  thread counts of every parallel stage so it degrades to fewer workers instead of OOMing.
- Junction geodesics run on the **eroded‑mito surface mesh** (`bioimage_cpp.distance.geodesic_distances_mesh`),
  so cost scales with surface vertices rather than a full‑grid array. This replaced earlier
  full‑grid graph approaches (a membrane‑voxel Dijkstra graph, and before that `MCP_Geometric`) that
  allocated arrays proportional to the bounding box and OOMed on large mitochondria.
- Membrane approximation is bbox‑cropped, empty‑slice‑skipped, and per‑slice parallel.

## Key interpretation caveats
- `crista_orientation_anisotropy` is rotation‑invariant (magnitude only).
- Junctions = literal crista∩membrane overlap; their number depends on the membrane band width
  (`mm_thickness`) and the voxel size.
- `crista_to_mito_surface_ratio` can be > 1 (folded cristae).
- `junction_clustering_index` assumes a flat‑surface CSR reference (approximation).
- Getting the voxel size right (nm, not Å) is essential for all physical quantities.
