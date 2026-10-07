# Volume EM mitochondria: the baselines at the voxel sizes selected on the validation blocks

Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.
`dice` is the semantic foreground dice. `val_metric` is the best validation DiceLoss of the run,
read from its checkpoint, where lower is better.

All models were segmented with identical preprocessing and identical post-processing
(seed_distance 1, boundary_threshold 0.12, area_threshold 200, min_size 1000).

Segmentations passed with `-s` were produced elsewhere and are only scored here, so the
sentence above does not apply to them. They were made with:

- `microsam-A-z25-xy5-xy`: ais_min_size 0, blocks /mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split, boundary_distance_threshold 0.5, canvas_shape 128/3200/3200, center_distance_threshold 0.5, distance_smoothing 1.6, foreground_smoothing 1.0, foreground_threshold 0.5, gap_closing 2, halo 128/128, inference_voxel_size_nm 25.0/5.0/5.0, invert False, micro_sam 1.8.14, min_size 1000, min_z_extent 2, mode xy slices merged in z, model vit_b_em_organelles, n_lost_to_native 0, sam_input 1024, segmentation_mode ais, target_voxel_size_nm 25.0/5.0/5.0, tile_shape 768/768
- `mitonet-A-z25-xy10-xy`: blocks /mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split, confidence_thr 0.3, empanada_napari 1.2.1, inference_voxel_size_nm 25.0/10.0/10.0, invert False, label_divisor 1000, median_kernel_size 5, min_extent 4, min_size 250, mode xy stack, model MitoNet_v1, n_lost_to_native 0, nms_kernel 3, nms_threshold 0.1, target_voxel_size_nm 25.0/10.0/10.0
- `mitonet-B-iso12.5-ortho`: allow_one_view False, blocks /mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split, cluster_iou_thr 0.75, confidence_thr 0.3, empanada_napari 1.2.1, inference_voxel_size_nm 12.5/12.5/12.5, invert False, label_divisor 1000, median_kernel_size 5, min_extent 4, min_size 320, mode ortho-plane consensus of xy, xz, yz, model MitoNet_v1, n_lost_to_native 0, nms_kernel 3, nms_threshold 0.1, pixel_vote_thr 2, target_voxel_size_nm 12.5/12.5/12.5
- `mitonet-C-z30-xy8-xy`: blocks /mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split, confidence_thr 0.3, empanada_napari 1.2.1, inference_voxel_size_nm 29.91/8.0/8.0, invert False, label_divisor 1000, median_kernel_size 5, min_extent 4, min_size 326, mode xy stack, model MitoNet_v1, n_lost_to_native 0, nms_kernel 3, nms_threshold 0.1, target_voxel_size_nm 30.0/8.0/8.0

## Averaged over the test blocks

| model                   | f1     | precision | recall | msa    | sbd    | dice   | val_metric | iteration |
|-------------------------|--------|-----------|--------|--------|--------|--------|------------|-----------|
| microsam-A-z25-xy5-xy   | 0.0734 | 0.0458    | 0.1845 | 0.0152 | 0.1117 | 0.3786 | -          | -         |
| mitonet-A-z25-xy10-xy   | 0.0938 | 0.0698    | 0.1430 | 0.0263 | 0.1665 | 0.3675 | -          | -         |
| mitonet-B-iso12.5-ortho | 0.1013 | 0.0828    | 0.1314 | 0.0186 | 0.1831 | 0.3281 | -          | -         |
| mitonet-C-z30-xy8-xy    | 0.1209 | 0.0867    | 0.1996 | 0.0267 | 0.1718 | 0.4073 | -          | -         |
| paper                   | 0.6905 | 0.5850    | 0.8469 | 0.3197 | 0.5722 | 0.8490 | 0.2927     | 14500     |

## Per test block

| model                   | dataset | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|-------------------------|---------|--------|-----------|--------|--------|--------|--------|--------|--------|
| microsam-A-z25-xy5-xy   | 4007    | 0.0247 | 0.0152    | 0.0667 | 0.0037 | 0.0657 | 0.2162 | 132    | 30     |
| mitonet-A-z25-xy10-xy   | 4007    | 0.0625 | 0.0455    | 0.1000 | 0.0172 | 0.1413 | 0.3691 | 66     | 30     |
| mitonet-B-iso12.5-ortho | 4007    | 0.0706 | 0.0545    | 0.1000 | 0.0134 | 0.1519 | 0.3225 | 55     | 30     |
| mitonet-C-z30-xy8-xy    | 4007    | 0.1010 | 0.0725    | 0.1667 | 0.0231 | 0.1587 | 0.4163 | 69     | 30     |
| paper                   | 4007    | 0.6410 | 0.5208    | 0.8333 | 0.3196 | 0.5188 | 0.8742 | 48     | 30     |
| microsam-A-z25-xy5-xy   | 4009    | 0.1221 | 0.0765    | 0.3023 | 0.0266 | 0.1577 | 0.5410 | 170    | 43     |
| mitonet-A-z25-xy10-xy   | 4009    | 0.1250 | 0.0941    | 0.1860 | 0.0355 | 0.1918 | 0.3658 | 85     | 43     |
| mitonet-B-iso12.5-ortho | 4009    | 0.1321 | 0.1111    | 0.1628 | 0.0239 | 0.2143 | 0.3338 | 63     | 43     |
| mitonet-C-z30-xy8-xy    | 4009    | 0.1408 | 0.1010    | 0.2326 | 0.0304 | 0.1848 | 0.3982 | 99     | 43     |
| paper                   | 4009    | 0.7400 | 0.6491    | 0.8605 | 0.3198 | 0.6256 | 0.8237 | 57     | 43     |
