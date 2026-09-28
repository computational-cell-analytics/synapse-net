# Volume EM mitochondria

Training, inference and evaluation for the volume EM (FIB-SEM) mitochondria model, at a voxel size of
25 nm in z and 5 nm in xy. The recipe itself lives in `synapse_net.training.mitochondria_vol_em`; the
scripts here are the cluster-specific parts and the reproduction of the published model.

```
training/
  train_mitochondria_vol_em.py              the published recipe, with its constants
  split-mito_vol_em_aniso2lvl_final.json    the pinned split of the published run, 14 train / 2 val
  compare_training_setup.py                 checks that the port builds the published training setup
  repair_checkpoint.py                      makes the published checkpoint loadable again
  repro_aniso2lvl_seed{42,43,44}.yaml       job manifests for the reproduction runs
inference/
  run_mitochondria_vol_em_segmentation.py   segmentation with a local checkpoint
evaluation/
  compare_models.py                         model comparison on the held-out test blocks
  voxel_sweep.py                            block lookup and resampling shared by the baseline sweeps
  segment_mitonet.py                        MitoNet baseline at a chosen voxel size, for compare_models.py -s
  mitonet_iso30.yaml                        job manifest for it at 30 nm isotropic on the test blocks
  mitonet_sweep_val.yaml                    job manifest for the MitoNet sweep on the validation blocks
  mitonet_sweep_score.yaml                  job manifest for scoring it
  segment_microsam.py                       micro-sam baseline at a chosen voxel size, for compare_models.py -s
  microsam_sweep_val.yaml                   job manifest for the micro-sam sweep on the validation blocks
  microsam_sweep_score.yaml                 job manifest for scoring it
  microsam_test_selected.yaml               job manifest for the test run of the selected setting
  score_voxel_size_sweep.py                 scores a voxel-size sweep of one method, selects per approach
```

## The published model

`/mnt/lustre-grete/usr/u15205/volume-em/models/checkpoints/volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final`,
best validation DiceLoss **0.292714**.

Two things about it are not visible from its configuration:

- **It cannot be loaded.** torch-em pickles the training dataset into the checkpoint, and the script
  that trained it defined its raw transform as a local function, so the checkpoint references
  `__main__._raw_transform_with_fix_white_patches`. Use `training/repair_checkpoint.py` to write a
  loadable copy; the weights are copied through bit-identically and the original is not modified.
- **Its counters describe only half of its history.** Two jobs were submitted 24 minutes apart and ran
  concurrently for ~23 h against the same checkpoint folder. The second warm-started weights-only from
  the first's `best.pt` as it stood at epoch 15, so its optimizer and counters restarted. Both ended on
  early stopping, neither on the configured 50,000 iterations nor on the wall clock. The surviving
  `best.pt` is the second job's: ~16,500 gradient steps in total, with an optimizer reset at 2,000.

The recipe as documented is a single clean run, and that is what this module trains. Running twice
with the same name and `save_root` is now refused rather than silently racing.

## Reproduction

Three clean runs, seeds 42/43/44, into a separate `save_root`:

```bash
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py \
    scripts/volume_em/training/repro_aniso2lvl_seed42.yaml      # and 43, 44
```

Progress goes to **stderr**, so watch the `.err` file that `sbatch_runner.py` writes, not the `.log`:
the `.log` only holds the four startup lines, and looks stalled if you watch it instead.

```bash
tr '\r' '\n' < logs/sbatch_jobs/<manifest>_<timestamp>.err | tail -3
```

`synapse_net` is installed editable, so a queued job picks up whatever is in the working tree at the
moment it *starts*, not at the moment it was submitted. Do not switch branches in this checkout while
runs are queued or running, or they will train something other than what you submitted. `git log -1`
of the checkout at job start is worth recording alongside the results.

Before spending the GPU time, the cheap check is the one that actually verifies the port:

```bash
python scripts/volume_em/training/compare_training_setup.py
```

It builds the trainer through the ported recipe, stops before training, and diffs it against the
setup stored in the published checkpoint — model kwargs, optimizer, scheduler, loss, both loader
kwargs, the sampler parameters, the per-file dataset lengths and the file lists. It also checks the
two places where the port deviates structurally: the raw transform is compared numerically against a
transcription of the published one, and the `PadIfNecessary` that the port adds to the joint transform
is confirmed never to fire.

### What to expect

- **Stopping.** `early_stopping=20` is the operative criterion, not `n_iterations`. Every historical
  run stopped after 16k–19k iterations / 26–29 h with 8–10 h of its limit unused. A forced 50,000
  iterations would take ~71 h at the measured 5.13 s/it, which exceeds the 48 h partition maximum.
- **Run-to-run spread.** Three comparable trainings on this split reached 0.292714, 0.296378 and
  0.298842, so a spread of about 0.006 is normal and a reproduction inside that band is
  indistinguishable from noise.
- **torch-em version.** The reproduction runs against the torch-em checkout that produced the
  published model (`f247988b`, 0.8.3), so the library is not a variable. The setup check also passes
  under a clean environment built from `environment.yaml` (Python 3.14, torch 2.13, torch_em 0.10.6):
  every recipe field is identical there too, and the only difference is a `mixed_precision_dtype`
  field that newer torch-em adds and that this recipe never reads, since it trains in single
  precision. The check reports such additions explicitly rather than ignoring unknown fields.
- **Seeding.** `--seed` makes runs repeatable in every way that matters — two runs with the same seed
  agree to ~1e-7 in the validation metric — but not bit-identically, because cudnn picks kernels
  adaptively. `--deterministic` does *not* fix that here (torch-em sets `warn_only=True`, so
  non-deterministic kernels silently fall back); it only costs throughput, so it is not used.

## Comparison

```bash
python scripts/volume_em/evaluation/compare_models.py \
    -m paper=/mnt/lustre-grete/usr/u15205/volume-em/models-repro/reference/volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final \
    -m seed42=/mnt/lustre-grete/usr/u15205/volume-em/models-repro/checkpoints/volume-em-mito-aniso2lvl-repro-seed42 \
    -m seed43=... -m seed44=... \
    -o scripts/volume_em/RESULTS.md
```

Every model is segmented with identical preprocessing and identical post-processing; tuning the
post-processing per model would fold a post-processing difference into what is meant to be a
comparison of the trained networks. Segmentations are cached, so a model can be added later without
redoing the others.

### Post-processing, and why `min_size` is not tuned here

The published post-processing parameters come from a grid search against a different, out-of-core
watershed, and under this one they over-segment: on the test blocks the model predicts 48 and 57
instances against 30 and 43 annotated ones, so precision (0.585) is far below recall (0.847) while the
semantic dice is a healthy 0.849. The extra objects are small fragments.

Raising `min_size` from 1,000 to 20,000 looks like a free fix — F1 goes from 0.69 to 0.85, the counts
become exactly 30/30 and 43/43, and recall does not move at all, so only false positives are removed.
**It is not a fix, and the default is deliberately left alone.** That threshold was read off the two
test blocks, whose mitochondria happen to be large (5th percentile 21,100 and 35,263 voxels). In the
14 training blocks, of 717 annotated mitochondria:

| below | objects | share |
|---|---|---|
| 1,000 voxels | 2 | 0.3 % |
| 5,000 voxels | 18 | 2.5 % |
| 10,000 voxels | 45 | 6.3 % |
| 20,000 voxels | 129 | **18.0 %** |

So a 20,000-voxel filter would throw away nearly a fifth of the objects the annotators marked. It
scores well on these two blocks and would generalize badly. The inherited `min_size=1000` is instead
consistent with the annotation: 99.7 % of annotated mitochondria are larger than it.

The real lever is the seed and boundary parameters, which is where the fragmentation comes from, and
re-tuning those needs a validation set that is not these two blocks. `compare_models.py
--size_filter_sweep` reproduces the table above as a diagnostic; read it as a measure of how much of
the error is fragments, not as a tuning result.

### MitoNet baseline

MitoNet (empanada, `MitoNet_v1`) is a 2D generalist whose 3D mode, the ortho-plane consensus, infers on
xy, xz and yz planes and so assumes isotropic voxels. `evaluation/segment_mitonet.py` resamples each
test block from 25/5/5 nm to 30 nm isotropic (107 x 267 x 267; linear with anti-aliasing), runs
ortho-plane inference, and resizes the result back to the native grid with nearest-neighbor
interpolation. The ground truth is never resampled. Both the consensus and the xy stack alone come
out of the same run:

```bash
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/mitonet_iso30.yaml
python scripts/volume_em/evaluation/compare_models.py \
    -m paper=/mnt/lustre-grete/usr/u15205/volume-em/models-repro/reference/volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final \
    -s mitonet-iso30-ortho=/mnt/lustre-grete/usr/u15205/volume-em/repro-comparison/mitonet-iso30-ortho \
    -s mitonet-iso30-xy=/mnt/lustre-grete/usr/u15205/volume-em/repro-comparison/mitonet-iso30-xy \
    --size_filter_sweep 1000 10000 20000 30000 \
    --title "Volume EM mitochondria: MitoNet at 30 nm isotropic vs. the published model" \
    -o scripts/volume_em/RESULTS_mitonet_iso30.md
```

It runs in its own environment, `/mnt/lustre-grete/usr/u15205/envs/empanada`, built from the synapse
repo's `env_empanada.yaml` (empanada-napari 1.2.1, torch 2.8).

- **Size filter.** `min_size` is 23 voxels at 30 nm, the same physical volume as synapse-net's 1,000
  native voxels (a 30 nm voxel is 43.2 native ones). empanada's default of 500 would be ~21,600
  native voxels, the kind of threshold the section above explains does not generalize, and it would
  flatter MitoNet on these two blocks. `min_extent` is empanada's 4.
- **Everything else is the `Engine3d` default**, as in the earlier MitoNet runs (`confidence_thr` 0.3,
  median filter over 5 slices). The napari widget's defaults differ (0.5, 3 slices, `min_extent` 5).
- **The raw is fed as-is.** There is no white-filler removal, which is part of synapse-net's
  preprocessing, not MitoNet's. The 4009 block has 11.4 % filler.
- **Result: MitoNet fails at 30 nm**, with F1 0 on both blocks in both modes
  (`RESULTS_mitonet_iso30.md`). On 4009 it predicts no objects at all. On 4007 its 16 (consensus)
  or 20 (xy) objects barely touch the mitochondria (semantic dice 0.01), and they are small.
  - **It is not a resampling artefact.** MitoNet's 2D engine run directly on the resampled slices
    also has a dice of 0, and no flip or transpose of the result improves the overlap.
  - **The cause is the scale, not the contrast.** The objects MitoNet predicts at 30 nm are dark
    (mean raw 97.5), while the mitochondria in these blocks are brighter than their surroundings
    (144 vs. 134 in 4007). That first suggested an inverted contrast. The voxel-size sweep below
    refutes it: the inverted contrast scores F1 ≈ 0 at every voxel size, while the raw contrast
    works once xy is at 8–15 nm.
- **The February 2026 MitoNet numbers are not comparable.** They are in
  `/mnt/lustre-grete/usr/u12103/mitochondria/volume-em/test_split_empanada_vs25-*` (F1 ≤ 0.07), and
  they differ in three ways:
  - they ran on anisotropic input, in the xy stack only;
  - they saved the raw panoptic stack as uint8, whose `1000 + n` IDs wrap (ID 1024 became background);
  - they were scored with border filtering (`eval_mitos_touching_borders.py -b 2 -z`) against
    downsampled labels.

### MitoNet voxel-size sweep

30 nm may simply be the wrong scale: a median mitochondrion here is 305–370 nm across (measured on
the validation blocks), so only ~10–12 px at 30 nm. The sweep looks for the voxel size, and the
contrast polarity, at which MitoNet works best.

**It is run on the two validation blocks of the pinned split, not on the test blocks.** MitoNet never
saw them, and they mirror the test pair: one 4007 and one 4009 block, `(128, 1600, 1600)`, 37 and 28
mitochondria, the 4009 one with 13.8 % filler. Only the setting selected for each approach is then
run once on the test split.

Every setting runs on the raw and on the inverted contrast (`255 - raw`):

| group | voxel size (z/y/x nm) | modes |
|---|---|---|
| A: z native, xy swept | 25 / {5, 7.5, 8, 10, 12.5, 15, 20, 25, 30} | xy stack |
| B: isotropic | {10, 12.5, 15, 20, 25, 30} in every axis | ortho-plane and xy stack |
| C: single settings | 30 / 8 / 8, which MitoNet_v1 was probably trained at | xy stack |

Why these bounds:
- **A below 5 nm** would upsample the native data.
- **B below 10 nm** would interpolate z by more than 2.5× and reach 0.5–1.6 G voxels.
- **Above 30 nm** the smallest mitochondria fall under ~8 px.

A setting at the edge of its grid is flagged by the scorer, as a cue to extend the grid by one step.
The sweep holds fixed:
- `min_size`, scaled to the same physical volume as synapse-net's 1,000 native voxels at every
  voxel size;
- `min_extent` at 4 voxels;
- the `Engine3d` and consensus defaults.

25 nm isotropic is shared by A and B. The xy stack comes out identical whether it runs alone or
next to the ortho-plane consensus, which was checked.

```bash
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/mitonet_sweep_val.yaml
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/mitonet_sweep_score.yaml \
    --dependency afterok:<sweep job id>
```

The scorer selects per group by the F1 averaged over the validation blocks, ties broken by msa.
It says plainly when even the best setting is below F1 0.05, rather than presenting a failure as a
selection. It writes `RESULTS_mitonet_sweep_val.md` and `best.json` in the sweep root, whose
`segment_args` are the arguments for the one test run. `compare_models.py --split_file
... --split_key val` scores models on the validation blocks the same way; their cached
segmentations go into a `val/` folder so they cannot be confused with the test ones.

**Result.** The val tables are in `RESULTS_mitonet_sweep_val.md`, and the test run of the three
selected settings is in `RESULTS_mitonet_voxel_size.md`.

- **Contrast.** The inverted contrast scores F1 ≈ 0 at every voxel size, so the raw contrast is
  the right one.
- **Scale.** With the raw contrast, MitoNet works only at fine scales. It peaks at 8–15 nm in xy
  (median mitochondrion 20–45 px across) and collapses to 0 at 25–30 nm, in every group.
- **Noise.** The top settings are within a few hundredths of each other, which is within noise
  for 65 objects.

| selected on val | val F1 | test F1 | test precision / recall | test dice |
|---|---|---|---|---|
| A: z 25, xy 10 nm, xy stack | 0.182 | 0.094 | 0.070 / 0.143 | 0.368 |
| B: 12.5 nm isotropic, ortho-plane | 0.235 | 0.101 | 0.083 / 0.131 | 0.328 |
| C: z 30, xy 8 nm, xy stack | 0.223 | 0.121 | 0.087 / 0.200 | 0.407 |
| synapse-net, published model | – | 0.691 | 0.585 / 0.847 | 0.849 |

At its best voxel size MitoNet goes from F1 0 to about 0.1 on the test blocks, still far below
synapse-net. It predicts 55–99 objects against 30 and 43, most of which do not match. The test
scores are about half the val scores, so the val selection should be read as "8–15 nm" rather
than as one exact voxel size.

### micro-sam voxel-size sweep

`evaluation/segment_microsam.py` runs the same grid with micro-sam's automatic instance segmentation
(`vit_b_em_organelles`, AIS decoder), selected on the same validation blocks by the same rule, with
the same physically matched size filter. It runs on the raw contrast only. It uses its own
environment, `/mnt/lustre-grete/usr/u15205/envs/micro-sam`, built from the micro-sam checkout's
`environment.yaml` plus an editable install of the checkout: micro-sam 1.8.14 (`de4231f`),
python-elf 0.9.2, bioimage-cpp 0.9.0.

- **The micro-sam version matters.** micro-sam 1.7.7 builds nifty graphs for the merge along z,
  while elf 0.9 takes bioimage-cpp graphs. Its `environment.yaml` only asks for `python-elf >=0.7.1`,
  so a fresh environment for 1.7.7 gets elf 0.9.2 and fails in the merge (`'UndirectedGraph' object
  has no attribute 'number_of_nodes'`). micro-sam 1.8.x requires elf 0.9 and works.
- The model is unchanged: the cached `vit_b_em_organelles` weights and decoder match the registry
  of 1.8.14.

- **The encoder scale is controlled, not only the resampling.**
  - SAM resizes every image, and every tile, so that its longest side is 1024 px, up or down.
    Passed as-is, a slice of these 8 µm blocks would reach the encoder at ~7.8 nm/px at every
    voxel size, only more or less blurred.
  - So every image or tile that holds data is made exactly 1024 px. A slice of up to 1024 px is
    reflect-padded to 1024 × 1024. The two larger ones, xy 5 and 7.5 nm, are padded onto a canvas
    on which micro-sam's tiling (768 + a halo of 128) gives every tile holding data its full 1024 px.
    That was checked for every grid size.
  - The padding is cropped away before scoring.
- **One mode.** micro-sam segments the xy slices and merges them along z; it has no ortho-plane
  mode.
- **Settings.** Everything else is micro-sam's own: the AIS defaults, and `gap_closing` 2 and
  `min_z_extent` 2, the napari annotator's defaults for automatic 3D segmentation.
- **The February 2026 `test_split_microsam_vs25-*` runs are not comparable.** They were never
  scored, and they are stored at the reduced grids. Because of the rescaling above, their "10" and
  "20 nm" runs saw ~7.8 nm/px.

```bash
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/microsam_sweep_val.yaml
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/microsam_sweep_score.yaml \
    --dependency afterok:<sweep job id>
python /mnt/vast-nhr/home/freckmann15/u15205/synapse/sbatch_runner.py scripts/volume_em/evaluation/microsam_test_selected.yaml
```

**Result on the validation blocks** (`RESULTS_microsam_sweep_val.md`):

- **Finer is better, down to the native resolution.** In group A the F1 rises steadily as xy gets
  finer: 0 at 12.5–30 nm, 0.03 at 10 nm, 0.04 at 7.5–8 nm, and **0.089 at the native 5 nm**.
  - The first run selected 5 nm at the edge of its grid, so the grid was extended by a step.
    At 3.75 nm, which upsamples the native data, the F1 falls back to 0.057, so 5 nm is a real peak.
  - At 5 nm there is no resampling at all, only the tiled 1024 px canvas.
- **B and C have no working setting.** The best isotropic setting is 10 nm at F1 0.017, and 30/8/8
  reaches 0.029, both below the floor of 0.05, so neither group has a selection.
  - B is also rising towards its finest setting. It could only be extended below 10 nm by
    interpolating z by more than 2.5×, and A shows that even the native data gives 0.089.
- **micro-sam over-segments at every scale that finds anything.** At 5 nm it predicts 346 objects
  against 65 (precision 0.05, recall 0.29). Its semantic dice of 0.46 shows it finds the
  mitochondria, but splits them into fragments that fail the IoU 0.5 match.
- **Compared with MitoNet** on the same blocks and grid, micro-sam is worse everywhere: MitoNet's
  best settings reach F1 0.18–0.23.

**Result on the test blocks** (`RESULTS_baselines_voxel_size.md`). The selected setting was run
once, and is scored next to the MitoNet settings selected the same way:

| setting (selected on val) | val F1 | test F1 | test precision / recall | test dice | objects 4007 / 4009 |
|---|---|---|---|---|---|
| micro-sam A: z 25, xy 5 nm (native) | 0.089 | 0.073 | 0.046 / 0.185 | 0.379 | 132 / 170 |
| MitoNet A: z 25, xy 10 nm | 0.182 | 0.094 | 0.070 / 0.143 | 0.368 | 66 / 85 |
| MitoNet B: 12.5 nm isotropic, ortho-plane | 0.235 | 0.101 | 0.083 / 0.131 | 0.328 | 55 / 63 |
| MitoNet C: z 30, xy 8 nm | 0.223 | 0.121 | 0.087 / 0.200 | 0.407 | 69 / 99 |
| synapse-net, published model | – | 0.691 | 0.585 / 0.847 | 0.849 | 48 / 57 |

The ground truth has 30 / 43 objects.

- **Both generalists fail on this data at every voxel size.** Each detects part of the mitochondria
  (dice 0.33–0.41) but splits it into fragments, which is why precision is so low.
- **micro-sam is worse than MitoNet**, with the gap largest on 4007 (F1 0.025 against 0.06–0.10).
- **The ranking holds on test.** It matches the one on val, which suggests the val selection
  carried over.

### Test data

The only held-out ground truth is two blocks under
`/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split/`:

| block | shape | instances | filler |
|---|---|---|---|
| `4007/4007_block_z001024_001152_y001936_003872_x003312_004968.h5` | (128, 1600, 1600) | 30 | none |
| `4009/4009_raw_z128-256_y1600-3200_x0-1600_6.h5` | (128, 1600, 1600) | 43 | 11.4 % |

Neither overlaps the training data: the 4007 block continues the training z-stack where it ends, the
4009 block is the one grid tile that training does not use, and the two training cutouts whose
filenames carry no coordinates were checked by content (best normalized cross-correlation 0.33,
against 0.79 for merely adjacent slices and 1.0 for a true match). The blocks differ in whether they
contain filler at all, which makes the pair a useful control, so report them separately as well as
averaged.

The `test_split_s1*`, `_s2_*` and `_s3_*` folders are the *same two blocks* downsampled, not extra
test volume, and most of them store `raw` as float64, which the filler removal correctly rejects.

## Known issue, not addressed here

Every inference and evaluation config in the `synapse` repo points at
`volume-em-mito-net32-lr1e-4-bs4-ps32x512x512-final`, a different architecture (one anisotropic level
rather than two). On 2026-05-28 a job auto-warm-started from it, ran 125 steps at a reset learning
rate of 1e-4 and was killed; its validation metric went 0.320197 → 0.394636 and the good February
weights are gone. Numbers produced through that path after 2026-05-28 12:37 come from a degraded
model. The recipe here does not expose `--scale_factors`, so it cannot currently retrain that variant.
