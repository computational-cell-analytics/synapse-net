import os
from typing import List, Optional

import h5py
import pandas as pd
import numpy as np
import vigra

from elf.evaluation.matching import _compute_scores, _compute_tps
from elf.evaluation import dice_score
from elf.segmentation.workflows import simple_multicut_workflow
from scipy.ndimage import binary_dilation, binary_closing, distance_transform_edt, binary_opening
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import relabel_sequential, watershed
from tqdm import tqdm


def _expand_seg(az, iterations):
    return binary_closing(binary_dilation(az, iterations=iterations), iterations=iterations)


def _crop(seg, gt, return_bb=False):
    bb_seg, bb_gt = np.where(seg), np.where(gt)

    # Handle empty segmentations.
    if bb_seg[0].size == 0:
        bb = tuple(slice(bgt.min(), bgt.max() + 1) for bseg, bgt in zip(bb_seg, bb_gt))
    else:
        bb = tuple(slice(
            min(bseg.min(), bgt.min()), max(bseg.max(), bgt.max()) + 1
        ) for bseg, bgt in zip(bb_seg, bb_gt))

    if return_bb:
        return seg[bb], gt[bb], bb
    else:
        return seg[bb], gt[bb]


def _postprocess(data, apply_cc, min_component_size, iterations=0):
    if iterations > 0:
        data = _expand_seg(data, iterations)
    if apply_cc:
        data = label(data)
        ids, sizes = np.unique(data, return_counts=True)
        filter_ids = ids[sizes < min_component_size]
        data[np.isin(data, filter_ids)] = 0
        data, _, _ = relabel_sequential(data)
    return data


def _single_az_evaluation(seg, gt, apply_cc, min_component_size, iterations, criterion):
    assert seg.shape == gt.shape, f"{seg.shape}, {gt.shape}"
    dice = dice_score(seg > 0, gt > 0)

    seg = _postprocess(seg, apply_cc, min_component_size, iterations=iterations)
    gt = _postprocess(gt, apply_cc, min_component_size=500)

    n_true, n_matched, n_pred, scores = _compute_scores(seg, gt, criterion=criterion, ignore_label=0)
    tp = _compute_tps(scores, n_matched, threshold=0.5)
    fp = n_pred - tp
    fn = n_true - tp

    return {"tp": tp, "fp": fp, "fn": fn, "dice": dice}


def az_evaluation(
    seg_paths: List[str],
    gt_paths: List[str],
    seg_key: str,
    gt_key: str,
    crop: bool = True,
    apply_cc: bool = True,
    min_component_size: int = 5000,
    iterations: int = 3,
    criterion: str = "iou",
    threshold: Optional[float] = None,
    **extra_cols
) -> pd.DataFrame:
    """Evaluate active zone segmentations against ground-truth annotations.

    This computes the dice score as well as false positives, false negatives and true positives
    for each segmented tomogram.

    Args:
        seg_paths: The filepaths to the segmentations, stored as hd5 files.
        gt_paths: The filepaths to the ground-truth annotatons, stored as hdf5 files.
        seg_key: The internal path to the data in the segmentation hdf5 file.
        gt_key: The internal path to the data in the ground-truth hdf5 file.
        crop: Whether to crop the segmentation and ground-truth to the bounding box.
        apply_cc: Whether to apply connected components before evaluation.
        min_component_size: Minimum component size for filtering the segmentation before evaluation.
        iterations: Post-processing iterations for expanding the AZ annotations.
        criterion: The criterion for matching annotations and segmentations
        threshold: Threshold applied to the segmentation. This is required if the segmentation is passed as
        probability prediction instead of a binary segmentation. Possible values: 'iou', 'iop', 'iot'.
        extra_cols: Additional columns for the result table.

    Returns:
        A data frame with the evaluation results per tomogram.
    """
    assert len(seg_paths) == len(gt_paths)

    results = {key: [] for key in extra_cols.keys()}
    results.update({
        "tomo_name": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "dice": [],
    })

    i = 0
    for seg_path, gt_path in tqdm(zip(seg_paths, gt_paths), total=len(seg_paths), desc="Run AZ Eval"):
        with h5py.File(seg_path, "r") as f:
            if seg_key not in f:
                print("Segmentation", seg_key, "could not be found in", seg_path)
                i += 1
                continue
            seg = f[seg_key][:].squeeze()

        with h5py.File(gt_path, "r") as f:
            gt = f[gt_key][:]

        if threshold is not None:
            seg = seg > threshold

        if crop:
            seg, gt = _crop(seg, gt)

        result = _single_az_evaluation(seg, gt, apply_cc, min_component_size, iterations, criterion=criterion)
        results["tomo_name"].append(os.path.basename(seg_path))
        for res in ("tp", "fp", "fn", "dice"):
            results[res].append(result[res])
        for name, val in extra_cols.items():
            results[name].append(val[i])
        i += 1

    return pd.DataFrame(results)


def _get_presynaptic_mask(boundary_map, vesicles):
    mask = np.zeros(vesicles.shape, dtype="bool")

    def _compute_mask_2d(z):
        distances = distance_transform_edt(boundary_map[z] < 0.25).astype("float32")
        seeds = vigra.analysis.localMaxima(distances, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
        seeds = label(np.isnan(seeds))
        overseg = watershed(boundary_map[z], markers=seeds)
        seg = simple_multicut_workflow(
            boundary_map[z], use_2dws=False, watershed=overseg, n_threads=1, beta=0.6
        )

        def n_vesicles(mask, seg):
            return len(np.unique(seg[mask])) - 1

        props = pd.DataFrame(regionprops_table(seg, vesicles[z], properties=["label"], extra_properties=[n_vesicles]))
        ids, n_ves = props.label.values, props.n_vesicles.values
        presyn_id = ids[np.argmax(n_ves)]

        mask[z] = seg == presyn_id

    for z in range(mask.shape[0]):
        _compute_mask_2d(z)

    mask = binary_opening(mask, iterations=5)

    return mask


def thin_az(
    az_segmentation: np.ndarray,
    boundary_map: np.typing.ArrayLike,
    vesicles: np.typing.ArrayLike,
    tomo: Optional[np.typing.ArrayLike] = None,
    presyn_dist: int = 6,
    min_thinning_size: int = 2500,
    post_closing: int = 2,
    check: bool = False,
) -> np.ndarray:
    """Thin the active zone annotations by restricting them to a certain distance from the presynaptic mask.

    Args:
        az_segmentation: The active zone annotations.
        boundary_map: The boundary / membrane predictions.
        vesicles: The vesicle segmentation.
        tomo: The tomogram data. Optional, will only be used for evaluation.
        presyn_dist: The maximal distance to the presynaptic compartment, which is used for thinning.
        min_thinning_size: The minimal size for a label component.
        post_closing: Closing iterations to apply to the AZ annotations after thinning.
        check: Whether to visually check the results.

    Returns:
        The thinned AZ annotations.
    """
    az_segmentation = label(az_segmentation)
    thinned_az = np.zeros(az_segmentation.shape, dtype="uint8")
    props = regionprops(az_segmentation)

    min_bb_shape = (32, 384, 384)

    for prop in props:
        az_id = prop.label

        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:3], prop.bbox[3:]))
        pad_width = [max(sh - (b.stop - b.start), 0) // 2 for b, sh in zip(bb, min_bb_shape)]
        bb = tuple(
            slice(max(b.start - pw, 0), min(b.stop + pw, sh)) for b, pw, sh in zip(bb, pad_width, az_segmentation.shape)
        )

        # If this is a small component then we discard it. This is likely some artifact in the ground-truth.
        if prop.area < min_thinning_size:
            continue

        # First, get everything for this bounding box.
        az_bb = (az_segmentation[bb] == az_id)
        vesicles_bb = vesicles[bb]
        # Skip if we don't have a vesicle.
        if vesicles[bb].max() == 0:
            continue

        mask_bb = _get_presynaptic_mask(boundary_map[bb], vesicles_bb)

        # Apply post-processing to filter out only the parts of the AZ close to the presynaptic mask.
        distances = np.stack([distance_transform_edt(mask_bb[z] == 0) for z in range(mask_bb.shape[0])])
        az_bb[distances > presyn_dist] = 0
        az_bb = np.logical_or(binary_closing(az_bb, iterations=post_closing), az_bb)

        if check:
            import napari
            tomo_bb = tomo[bb]

            v = napari.Viewer()
            v.add_image(tomo_bb)
            v.add_labels(az_bb.astype("uint8"), name="az-thinned")
            v.add_labels(az_segmentation[bb], name="az", visible=False)
            v.add_labels(mask_bb, visible=False)
            v.title = f"{prop.label}: {prop.area}"

            napari.run()

        thinned_az[bb][az_bb] = 1

    return thinned_az
