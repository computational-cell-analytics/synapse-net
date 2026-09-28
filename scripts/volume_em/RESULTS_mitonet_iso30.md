# Volume EM mitochondria: MitoNet at 30 nm isotropic vs. the published model

Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.
`dice` is the semantic foreground dice. `val_metric` is the best validation DiceLoss of the run,
read from its checkpoint, where lower is better.

All models were segmented with identical preprocessing and identical post-processing
(seed_distance 1, boundary_threshold 0.12, area_threshold 200, min_size 1000).

Segmentations passed with `-s` were produced elsewhere and are only scored here, so the
sentence above does not apply to them. They were made with:

- `mitonet-iso30-ortho`: allow_one_view False, cluster_iou_thr 0.75, confidence_thr 0.3, empanada_napari 1.2.1, inference_voxel_size_nm 29.91/29.96/29.96, label_divisor 1000, median_kernel_size 5, min_extent 4, min_size 23, mode ortho-plane consensus of xy, xz, yz, model MitoNet_v1, nms_kernel 3, nms_threshold 0.1, pixel_vote_thr 2
- `mitonet-iso30-xy`: confidence_thr 0.3, empanada_napari 1.2.1, inference_voxel_size_nm 29.91/29.96/29.96, label_divisor 1000, median_kernel_size 5, min_extent 4, min_size 23, mode xy stack, model MitoNet_v1, nms_kernel 3, nms_threshold 0.1

## Averaged over the test blocks

| model               | f1     | precision | recall | msa    | sbd    | dice   | val_metric | iteration |
|---------------------|--------|-----------|--------|--------|--------|--------|------------|-----------|
| mitonet-iso30-ortho | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0014 | 0.0054 | -          | -         |
| mitonet-iso30-xy    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0063 | 0.0036 | -          | -         |
| paper               | 0.6905 | 0.5850    | 0.8469 | 0.3197 | 0.5722 | 0.8490 | 0.2927     | 14500     |

## Per test block

| model               | dataset | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|---------------------|---------|--------|-----------|--------|--------|--------|--------|--------|--------|
| mitonet-iso30-ortho | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0028 | 0.0107 | 16     | 30     |
| mitonet-iso30-xy    | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0126 | 0.0072 | 20     | 30     |
| paper               | 4007    | 0.6410 | 0.5208    | 0.8333 | 0.3196 | 0.5188 | 0.8742 | 48     | 30     |
| mitonet-iso30-ortho | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-xy    | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0      | 43     |
| paper               | 4009    | 0.7400 | 0.6491    | 0.8605 | 0.3198 | 0.6256 | 0.8237 | 57     | 43     |

## Size filter sweep

Applied to the finished segmentations, which is not the same as running the watershed with
that `min_size`, so read it as a diagnostic for how much of the error is sub-threshold
fragments.

**Do not read a `min_size` off this table.** It would be tuning on the test data, and it
would generalize badly: these two test blocks happen to hold only large mitochondria, while
18% of the 717 mitochondria annotated in the training blocks are smaller than 20,000 voxels.
The default of 1,000 is the value consistent with the annotation, which has 99.7% of its
objects above it.

| model               | dataset | min_size | f1     | precision | recall | msa    | n_pred | n_true |
|---------------------|---------|----------|--------|-----------|--------|--------|--------|--------|
| paper               | 4007    | 1000     | 0.6410 | 0.5208    | 0.8333 | 0.3196 | 48     | 30     |
| paper               | 4007    | 10000    | 0.8065 | 0.7812    | 0.8333 | 0.4515 | 32     | 30     |
| paper               | 4007    | 20000    | 0.8333 | 0.8333    | 0.8333 | 0.4761 | 30     | 30     |
| paper               | 4007    | 30000    | 0.8136 | 0.8276    | 0.8000 | 0.4573 | 29     | 30     |
| paper               | 4009    | 1000     | 0.7400 | 0.6491    | 0.8605 | 0.3198 | 57     | 43     |
| paper               | 4009    | 10000    | 0.8132 | 0.7708    | 0.8605 | 0.3697 | 48     | 43     |
| paper               | 4009    | 20000    | 0.8605 | 0.8605    | 0.8605 | 0.4048 | 43     | 43     |
| paper               | 4009    | 30000    | 0.8571 | 0.8780    | 0.8372 | 0.4134 | 41     | 43     |
| mitonet-iso30-ortho | 4007    | 1000     | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 16     | 30     |
| mitonet-iso30-ortho | 4007    | 10000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 10     | 30     |
| mitonet-iso30-ortho | 4007    | 20000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 4      | 30     |
| mitonet-iso30-ortho | 4007    | 30000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 1      | 30     |
| mitonet-iso30-ortho | 4009    | 1000     | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-ortho | 4009    | 10000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-ortho | 4009    | 20000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-ortho | 4009    | 30000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-xy    | 4007    | 1000     | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 20     | 30     |
| mitonet-iso30-xy    | 4007    | 10000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 15     | 30     |
| mitonet-iso30-xy    | 4007    | 20000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 7      | 30     |
| mitonet-iso30-xy    | 4007    | 30000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 6      | 30     |
| mitonet-iso30-xy    | 4009    | 1000     | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-xy    | 4009    | 10000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-xy    | 4009    | 20000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
| mitonet-iso30-xy    | 4009    | 30000    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0      | 43     |
