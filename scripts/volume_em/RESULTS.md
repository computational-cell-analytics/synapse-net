# Volume EM mitochondria: reproduction vs. the published model

**Status: baseline only.** The three reproduction runs (seeds 42/43/44) are queued; regenerate
this file with `-m seed42=... -m seed43=... -m seed44=...` once they finish. The segmentations of
the published model are cached, so adding them does not redo this row.

Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.
`dice` is the semantic foreground dice. `val_metric` is the best validation DiceLoss of the run,
read from its checkpoint, where lower is better.

All models were segmented with identical preprocessing and identical post-processing
(seed_distance 1, boundary_threshold 0.12, area_threshold 200, min_size 1000).

## Averaged over the test blocks

| model | f1     | precision | recall | msa    | sbd    | dice   | val_metric | iteration |
|-------|--------|-----------|--------|--------|--------|--------|------------|-----------|
| paper | 0.6905 | 0.5850    | 0.8469 | 0.3197 | 0.5722 | 0.8490 | 0.2927     | 14500     |

## Per test block

| model | dataset | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|-------|---------|--------|-----------|--------|--------|--------|--------|--------|--------|
| paper | 4007    | 0.6410 | 0.5208    | 0.8333 | 0.3196 | 0.5188 | 0.8742 | 48     | 30     |
| paper | 4009    | 0.7400 | 0.6491    | 0.8605 | 0.3198 | 0.6256 | 0.8237 | 57     | 43     |
