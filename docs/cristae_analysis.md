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
    Skeleton-mode junctions are suppressed in the same zone (§4b), so on a thin volume this knob can
    gate a large fraction of the data — the preview line reports the percentage.
  - `junction_mode` (**Junction detection**) — `Overlap (crista ∩ membrane)` (default) or
    `Skeleton (crista reaching the membrane)`. See §4.
  - `max_extension` (**Max Extension**) — how far a crista may fall short of the membrane and still
    count (nm; `0` = use `mm_thickness`). Skeleton mode only.
  - `terminus_distance` (**Terminus Distance**) — how close a near‑membrane crista region must be to a
    crista terminus to count (nm; `0` = use the 20 nm default). Skeleton mode only.
  - `min_junction_volume` (**Min Junction Volume**) — junction regions smaller than this are discarded
    (nm³; `0` = use the 50 nm³ default). Skeleton mode only; removes specks, **not** the
    over‑detection described in §4b.
  - `show_membranes` (**Show Membrane Mesh**) — also add the eroded‑mito (lumen) inner surface — the
    surface the junction geodesics run along — as a **mesh** (napari Surface layer).
  - `show_skeleton` (**Show Crista Skeleton**) — add the crista centerline as a napari **Vectors** layer
    (`Crista Skeleton`, one line segment per skeleton edge, so it reads as a curve rather than as a
    point cloud) plus a **Points** layer `Crista Skeleton Termini` (the free ends). Both are restricted
    to `mito_seg > 0`, matching what the analysis actually skeletonises — previously the layer covered
    cristae outside every mitochondrion, which the detector never looks at. Skeleton mode's terminus
    filter keys on exactly those termini, so this is how you check whether a flagged junction sits at a
    real crista end or on a flank. Works in both junction modes.
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
| `--border_gap` | thickness | Distance from volume faces where the membrane is suppressed (nm). Skeleton-mode junctions are suppressed in the same zone. |
| `--method` | `skip` | Orientation anisotropy mode: `skip` / `fast` / `exact`. |
| `--membrane_mode` | `slice_2d` | Membrane shell construction: `slice_2d` / `shell_3d`. |
| `--junction_mode` | `overlap` | Junction detector: `overlap` (crista ∩ membrane) / `skeleton` (crista regions reaching within `--max_extension` of the membrane near a terminus; 3D only). See §4. |
| `--max_extension` | thickness | How far (nm) a crista may fall short of the membrane (`--junction_mode skeleton` only). |
| `--terminus_distance` | `20.0` | How close (nm) a near-membrane crista region must be to a crista terminus to count (`--junction_mode skeleton` only). |
| `--min_junction_volume` | `50.0` | Smallest junction volume (nm³) that counts (`--junction_mode skeleton` only). Removes specks only. |
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

### 4. Crista‑membrane junctions — `detect_junctions`
Two detectors are available, chosen with **Junction detection** in the widget / `--junction_mode` on
the command line. Both return `contact_voxel_count`, `crista_junction_count`, `contact_volume_nm3`,
`mean_junction_extension_nm`, and a labelled junction array (unique ID per junction), so everything
downstream — the junction geodesics of step 5 and the napari junction layer — is identical either way.
**Only `crista_junction_count` and `mean_junction_extension_nm` depend on the mode**;
`contact_voxel_count` and `contact_volume_nm3` always describe the genuine overlap, so they stay
comparable across runs.

#### 4a. `"overlap"` (default) — `detect_contact_sites`
A junction is the **pure intersection** of the crista mask and the membrane band
(`crista & membrane`) — **no hidden dilation/erosion**, so junctions correspond exactly to the
visible overlap of the two layers. Overlaps are grouped into discrete junctions by
**26‑connectivity connected components** (`scipy.ndimage.label`). `mean_junction_extension_nm` is
`NaN` — this mode measures no gap.
> Because the junction reach equals the membrane band width (`mm_thickness`), a too‑large thickness
> at fine voxel sizes flags cristae that merely come near the boundary.

#### 4b. `"skeleton"` — `detect_junctions_skeleton`
The gap-tolerant alternative, for the case where a pure intersection misses a real junction: a crista
segmented a few nm short of the inner boundary membrane has **no overlap at all** and so scores zero
under `"overlap"`.

A junction is one **26-connected component of `crista & (distance_to_reference <= max_extension)`**.
A TEASAR skeleton (`bioimage_cpp.skeleton.teasar`) then gates each region on lying within
`terminus_distance` of a crista **terminus** (a degree-1 skeleton end), which is what separates a
crista *ending* at the membrane from one merely *running alongside* it.

**The reference surface is the inner boundary membrane, not the membrane band.** The band is several nm
thick and `distance_transform(~band)` is *identically 0* on every voxel of it, so it cannot say where
within the band a contact sits — verified on a synthetic crista penetrating an 8 nm band: all 200 of its
in-band voxels read 0.0, while their distances to the inner surface spread over 1.50–7.50 nm. So when a
lumen is available (every production caller supplies one) the distance is measured to
`lumen & ~erode(lumen)`, the outermost lumen layer — the same single-wall surface the junction geodesics
already run along (§5). Without a lumen the function falls back to the band, which is what the unit
tests exercise.

Two consequences worth knowing:
- The candidate set barely changes (400 → 440 voxels on the synthetic test — slightly *larger*, since
  the inner surface sits one voxel further in than the band). **This change does not reduce
  `crista_junction_count`**; it makes each junction locatable.
- `max_extension` is now measured from the inner boundary surface, so a crista lying deeper into the
  band than `max_extension` — out toward the outer membrane — is excluded where the band reference
  always accepted it. Rare in practice.

**The label is the closest-approach footprint, not the whole region.** The candidate region is a slab of
crista running anywhere within `max_extension` of the membrane, so labelling all of it produced a fat
blob whose centre of mass sat in the middle of the crista rather than at the contact. The label is now
`region & (distance <= gap + footprint_nm)` with `footprint_nm` defaulting to one voxel diagonal.
Measured on `cutout_mito2`: median junction label 2035 → 212 voxels (**9.6× smaller**) and the label
centroid moved from 4.67 nm to 1.74 nm from the membrane band. Every footprint still contains the
region's closest voxel, and the footprint's distance span is far tighter than the region's (0–2.13 nm
against 0–8.77 nm on one real junction).

Reported per junction: `mean_junction_extension_nm`, the region's closest approach to the reference
surface — 0 where the crista reaches it. Expect 0 to a few nm; a value sitting at `max_extension` means
the tolerance is doing the work rather than the geometry.

Because a junction is a subset of the crista mask, **two disconnected cristae can never be fused into
one junction**, and no parameter governs that. The footprint is a subset of the crista, hence always
inside the mitochondrion.

- **The volume‑border zone is excluded, to match overlap mode.** `approximate_membrane` deletes the
  membrane within `border_gap` of any volume face because the segmentation is cut off there and
  membrane presence is **unknown**, not absent. Overlap mode inherits that for free — `crista ∩
  membrane` cannot fire where the membrane is zero. A proximity test does not: the distance transform
  will measure straight across the deleted region to the nearest *surviving* membrane voxel and assert
  a junction against a membrane it was told nothing about. So the candidate mask has the border zone
  removed. Measured on real exports, junction voxels inside the zone went 2505 → 0 and 991 → 0.
  Regions are **trimmed, not discarded**: a crista entering the unknown zone still counts wherever
  else it genuinely reaches the membrane, exactly as under overlap.
  > **Watch this on thin slabs.** The zone is `border_gap` deep on *every* face, so on a 39‑slice
  > tomogram at 0.87 nm voxels the default 8 nm gap declares **48% of the volume unknown**, and the
  > junction count there fell from 12 to 10 once the zone was respected. The widget's preview line
  > reports the border‑zone percentage for this reason; lower `border_gap` if it is dominating.

> ### ⚠️ This mode over‑detects on densely packed cristae
>
> Proximity is not the same as junction. On a real mitochondrion with many cristae
> (`TS_PS_01` mito 1, 0.8681 nm voxels, 8 nm membrane) skeleton mode reports **21 junctions of which
> only 2 involve any literal crista–membrane contact**. The other 19 are cristae merely *passing
> within 8 nm* of the inner boundary membrane. This is a property of the premise, not a bug: in a
> dense mitochondrion "a crista comes within 8 nm of the membrane" is a common condition that mostly
> does not mean a junction is there.
>
> Regions kept at each threshold, `n (m)` = regions (of which contain a literal contact):
>
> | `max_extension` | no min | ≥50 nm³ | ≥200 nm³ | ≥500 nm³ | ≥2000 nm³ |
> |---|---|---|---|---|---|
> | 2 nm | 10 (2) | 3 (2) | 0 (0) | 0 (0) | 0 (0) |
> | 4 nm | 16 (2) | 9 (2) | 5 (2) | 0 (0) | 0 (0) |
> | **8 nm (default)** | **21 (2)** | 16 (2) | 13 (2) | 9 (2) | 0 (0) |
> | 12 nm | 25 (2) | 20 (2) | 18 (2) | 15 (2) | 9 (2) |
>
> The false positives are full‑sized (up to ~1750 nm³), so `min_junction_volume` cannot remove them
> without removing the real junctions too. **Five discriminators have been measured against real data and
> none separates the two populations** — do not re‑propose them without new evidence:
>
> | discriminator | why it fails |
> |---|---|
> | region elongation | real junctions are 1.5–2.4 elongated too |
> | region axis vs. membrane normal | 75–90° for *every* region: a contact patch spreads along the membrane by nature |
> | rate at which membrane distance drops toward the terminus | 0.42–0.73 for every region |
> | minimum region size | see the grid above |
> | **crista sheet normal vs. membrane normal** | **inverts on real data** — see below |
>
> The last one was implemented, measured and removed, and is worth recording in full because it looks
> compelling on paper. A crista is a lamella, so compare its *sheet* normal (gradient of the mask
> smoothed at the sheet thickness) to the membrane normal (gradient of the reference distance field): a
> crista running parallel to the membrane should score `|cos| → 1`, one meeting it end-on `|cos| → 0`.
> Note this is **not** the "region axis vs. membrane normal" row above — a contact patch's long axis is
> ~90° from the normal for any patch at all, whereas a sheet normal is not constrained that way.
>
> On synthetic geometry it behaved exactly as predicted, 0.00 for a sheet ending on a wall against 0.70
> for one running alongside. On `cutout_mito2`, whose three crista–membrane contacts are hand-verified,
> it **inverted**:
>
> | region | closest approach | `\|cos\|` at that voxel | literal contact? |
> |---|---|---|---|
> | 2 | 0.00 nm | 0.881 | **yes** |
> | 4 | 0.00 nm | 0.786 | **yes** |
> | 3 | 5.56 nm | 0.766 | no |
> | 1 | 0.00 nm | 0.000 | no |
>
> A threshold of 0.7 removes all three real junctions and keeps the false positive. Region means do not
> separate either (0.582/0.606 for contacting against 0.558/0.842 for non-contacting). Part of the cause
> is that the measure is ill-posed exactly where it has to be sampled: at a closest-approach voxel the
> distance field can be locally flat, and a vanishing gradient normalises to a meaningless direction —
> the exact 0.000 above is that artefact. Reproduce with
> `scripts/cooper/measure_terminus_alignment.py`, which is kept as the evidence and as a harness for
> testing the next candidate against the same ground truth.
>
> **What to do.** Treat `crista_junction_count` from this mode as an **upper bound** on a dense
> mitochondrion and inspect the result — enable **Show Crista Skeleton** and check whether each
> flagged region sits at a crista terminus or on a flank. Use `"overlap"` when only junctions with
> actual membrane contact should count; that is the one signal that separates cleanly here (the
> "real" column is exactly 2 at every setting above).

> **Why not extend the skeleton along its tangent?** That was the original design and it was measured
> to fail, so it is worth recording. On a real mitochondrion with three junctions, the direction from
> each skeleton end to its nearest actual contact was **142–146° away from that end's tangent** —
> pointing backwards — for all three. Tangent extension found no gap-bridged junction at any threshold
> up to 20 nm, because only 3 of 71 skeleton ends were within 8 nm of the membrane: TEASAR's medial
> axis stops well short of the rim. Widening the ray to a ±60° cone changed nothing; only an
> omnidirectional search found all three, which is a proximity test in disguise. The reason is
> geometric — a crista–membrane contact is a **rim** feature while a skeleton end is a **centerline**
> feature, and for a sheet meeting the membrane obliquely their directions are unrelated.
>
> **Why not group per-endpoint hits with a merge radius?** Also measured to fail. On the same
> mitochondrion no radius returns the correct three: the count steps 10, 8, 4, 2 as the radius grows,
> because a small radius fragments one junction while a large one fuses two that are 13.9 nm apart.
> Junction identity has to come from the contact geometry, which is what the connected-component
> definition above does.

#### 4c. The crista skeleton and its termini — `compute_crista_skeleton`

Raw TEASAR output is not usable as a set of crista ends, and this is what the terminus filter and the
**Show Crista Skeleton** layers both depend on. TEASAR spans a lamella with a *caterpillar* — a main
path plus many short side branches whose tips line the sheet rim — every segmentation speck contributes
its own miniature skeleton, and counting `degree <= 1` also counts isolated vertices. On a
tomogram-scale 40-lamella mask that gives **10080 termini** where roughly 80 are real, which is why the
skeleton and termini layers looked like confetti.

Three cleanup steps, all keyed on arclength in nm so they do not depend on the voxel size:

| step | knob | effect |
|---|---|---|
| drop speck components (total skeleton length below the threshold, and isolated vertices) | `min_skeleton_nm`, default 10 nm | a mask of pure specks yields an empty skeleton |
| collapse each fan of nearby termini **on the same component** to one representative | `terminus_merge_nm`, default 4 nm | 10080 → 480 termini on the 40-lamella mask |
| count `degree == 1`, not `degree <= 1` | — | an isolated vertex is not the free end of anything |

The representative kept is the cluster member nearest the cluster centroid — a real skeleton vertex, not
the centroid itself, so a terminus always lies on the skeleton and inside the crista.

Two things here are easy to get wrong and are pinned by tests:

- **Merging must not cross skeleton components.** Densely packed cristae sit a few nm apart, so plain
  single-linkage clustering chains termini between *different* cristae. On a 40-lamella mask at 6 nm
  spacing an 8 nm radius gives 5 clusters for the whole volume unrestricted, against 200 restricted to
  one component — same radius, same data.
- **No leaf-branch pruning.** Pruning short spurs is the obvious first idea and it was measured to be
  actively harmful: at a lamella's real end the main path arrives as a short leaf off a nearby branch
  node, so a length threshold deletes the real crista end. On the test lamella spanning y 13–46 it left
  termini only at y 45–46 and lost the y=13 junction entirely, and by fragmenting components it left
  *fewer* termini mergeable afterwards (960 against 480 on the same mask). Component filtering plus the
  per-component merge is both simpler and strictly better.

`terminus_merge_nm` is a genuine trade-off, not a free tidy-up: the detector accepts a region only if it
lies within `terminus_distance` of *some* terminus, so collapsing a whole sheet rim to one point costs
real junctions elsewhere on that rim. 16 nm reaches the cosmetic ideal of two termini per lamella and
reduces a small sheet to a single terminus — which is that failure. The 4 nm default is the scale of the
ragged fan at one end, not of the crista.

Because a crista shorter than `min_skeleton_nm` has no skeleton, it contributes no terminus and
therefore **cannot score a skeleton-mode junction** at all. That is the second, independent route by
which a speck is rejected, alongside `min_junction_volume`.

> `max_extension` is the tolerance for the crista stopping short of the membrane, so set it to about
> one membrane thickness. Pushed well above that it starts counting cristae that merely *pass near*
> the boundary, and it also begins fusing junctions whose near-membrane regions touch. Raise
> `min_extension_nm` above 0 to isolate junctions that genuinely span a gap, excluding those already
> touching the band. `terminus_distance` defaults to 20 nm — generous by design; the measured terminus
> distances of real junctions were 0.0, 0.0 and 8.6 nm. On every dataset tested the filter is a no-op,
> and it is retained for the tangential-crista case rather than because it is measured to help.

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
| Junction detector dispatch (`overlap` / `skeleton`) | `detect_junctions` | `synapse_net/cristae_analysis.py` |
| Junctions, `overlap` (crista ∩ membrane) | `detect_contact_sites` | `synapse_net/cristae_analysis.py` |
| Junctions, `skeleton` (near-membrane crista regions) | `detect_junctions_skeleton` → `_inner_surface_distance` (inner boundary membrane reference), `bioimage_cpp.distance.distance_transform`, `_border_zone` (volume-border exclusion), `compute_crista_skeleton` (terminus filter) | `synapse_net/cristae_analysis.py` |
| Crista skeleton & termini (also the **Show Crista Skeleton** layers) | `compute_crista_skeleton` → `bioimage_cpp.skeleton.teasar`, `_skeleton_graph` (speck-component removal, `networkx`), `_merge_termini` (per-component terminus clustering) | `synapse_net/cristae_analysis.py` |
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
| `crista_junction_count` | count | **Depends on `junction_mode`.** `overlap`: number of 26‑connected components of the contact voxels. `skeleton`: number of 26‑connected crista regions within `max_extension` of the **inner boundary membrane surface**, near a terminus of the cleaned‑up crista skeleton. | `overlap`: one junction = one connected crista–membrane contact patch, so a crista stopping short of the membrane gives 0 and a rim can fragment. `skeleton`: gap‑tolerant, and two disconnected cristae can never be counted as one junction. A crista shorter than `min_skeleton_nm` has no terminus and cannot count at all. See §4. |
| `contact_volume_nm3` | nm³ | `contact_voxel_count` × voxel volume. | — |
| `mean_junction_extension_nm` | nm | Mean over junctions of each junction region's **closest approach** to the **inner boundary membrane surface** (§4b), or to the membrane band when no lumen is supplied. | `NaN` unless `junction_mode="skeleton"`. Sanity check: expect 0 to a few nm; a value sitting at `max_extension` means the tolerance is doing the work, not the geometry. `0` means the crista reaches the inner boundary membrane. **Not signed** — a crista pushing into the band reads the same as one stopping short of the surface. |
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
  surface geodesics, `bioimage_cpp.filters.structure_tensor_eigenvalues` for the crista orientation
  anisotropy, and `bioimage_cpp.skeleton.teasar` for the `skeleton` junction mode (3D TEASAR; returns
  a vertex/edge centerline graph in nm, not a raster skeleton).
- **scikit‑image** — `measure.marching_cubes`, `measure.mesh_surface_area`, `measure.regionprops`;
  `morphology.ball`, `morphology.disk`, `morphology.local_maxima`.
- **networkx** — the crista skeleton graph: speck-component removal and per-component terminus
  clustering (`_skeleton_graph`, `_merge_termini`). Imported lazily inside those two helpers.
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
- `junction_mode="skeleton"` costs **one extra distance transform per mitochondrion** — the inner
  boundary membrane reference (§4b) on top of the membrane‑band transform the proximity stage needs.
  The skeleton graph cleanup adds ~1.5 s on a 243 k‑vertex skeleton against TEASAR's own ~1.7 s.

## Key interpretation caveats
- `crista_orientation_anisotropy` is rotation‑invariant (magnitude only).
- `crista_junction_count` depends on `junction_mode`, and the two modes are **not** interchangeable —
  do not pool or compare counts across modes. `overlap` is the literal crista∩membrane overlap, so its
  number depends on the membrane band width (`mm_thickness`) and the voxel size, and it reads 0 when
  the crista segmentation stops short of the membrane. `skeleton` counts near-membrane crista regions
  instead, so it depends on `max_extension` (and on `terminus_distance` / `min_junction_volume` /
  `min_skeleton_nm` / `terminus_merge_nm`); keep those fixed across runs you compare. It
  **over-counts on densely packed cristae** (21 vs 2 on one real mitochondrion; see §4b). Check
  `mean_junction_extension_nm` when using it: a value sitting at `max_extension` rather than at 0 to a
  few nm means the tolerance is doing the work, not the geometry.
- **"`skeleton` is a superset of `overlap`" is measured, not guaranteed.** It used to follow by
  construction. It no longer does, for two reasons: a crista lying deeper into the band than
  `max_extension` is not a candidate against the inner-surface reference, and a crista whose skeleton is
  shorter than `min_skeleton_nm` has no terminus to gate on. On `cutout_mito2` skeleton mode still covers
  all three literal contacts (verified in `test_skeleton_mode_finds_every_contact`), but if you see a
  count *below* overlap mode's, that is real information about the segmentation rather than a bug.
- `crista_to_mito_surface_ratio` can be > 1 (folded cristae).
- `junction_clustering_index` assumes a flat‑surface CSR reference (approximation).
- Getting the voxel size right (nm, not Å) is essential for all physical quantities.
