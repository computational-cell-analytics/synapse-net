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
