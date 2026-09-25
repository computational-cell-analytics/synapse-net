"""Evaluate the cristae segmentation on the held-out test set.

This reproduces the evaluation of the published cristae models. The segmentation is scored voxel-wise
inside the mitochondria that carry cristae annotations, i.e. where the mitochondria state is 1, so the
background and the mitochondria without annotations (state 2) are ignored.

If the segmentation files also contain the foreground prediction ('pred/foreground', written by
segment_test_set.py with '--save_predictions'), the average precision (AP) and ROC-AUC of the prediction
are computed as well: in the whole annotated region ('all'), in a shell of the given thickness at the
mitochondria membrane ('band<t>', the region of the cristae junctions), and in the rest of it ('core<t>').

The results are written to 'evaluation_<model_name>.csv' (dice, precision and recall per tomogram) and
'evaluation_<model_name>_ap.csv' (AP and ROC-AUC per tomogram and region), each with the average over the
tomograms in the rows for 'all'.
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from elf.io import open_file
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score, roc_auc_score

from synapse_net.training.cristae import get_cristae_test_paths
from synapse_net.training.transform import CRISTAE_VOXEL_SIZE
from train_cristae import EXCLUDE, SPLIT_FILE, TRAIN_ROOTS

# The AP and ROC-AUC are computed on a random subset of at most this many voxels, to keep them fast.
MAX_POINTS_AP = 3_000_000
EPS = 1e-8


def evaluate_segmentation(labels, seg, state):
    """Compute the voxel-wise dice, precision and recall inside the annotated mitochondria."""
    region = state == 1
    labels = labels.astype(bool) & region
    seg = seg.astype(bool) & region
    tp = int(np.logical_and(seg, labels).sum())
    fp = int(np.logical_and(seg, ~labels).sum())
    fn = int(np.logical_and(~seg, labels).sum())
    return {
        "dice": float(2 * tp / (2 * tp + fp + fn + EPS)),
        "precision": float(tp / (tp + fp + EPS)),
        "recall": float(tp / (tp + fn + EPS)),
        "tp": tp, "fp": fp, "fn": fn,
        "pred_fg": int(seg.sum()), "gt_fg": int(labels.sum()), "eval_voxels": int(region.sum()),
    }


def _subsample(labels, pred, max_points):
    if labels.size <= max_points:
        return labels, pred
    index = np.random.default_rng(0).choice(labels.size, size=max_points, replace=False)
    return labels[index], pred[index]


def evaluate_prediction(labels, pred, state):
    """Compute the AP and ROC-AUC of the foreground prediction inside the annotated mitochondria.

    Also reports the mean prediction in and outside of the cristae, and in the mitochondria without
    annotations (state 2), where the network is not trained.
    """
    s1, s2 = state == 1, state == 2
    gt1 = labels[s1] > 0
    fg1 = pred[s1]
    scores = {
        "n_state1": int(s1.sum()),
        "gt_fg": int(gt1.sum()),
        "n_state2": int(s2.sum()),
        "fg_mean_gtpos_state1": float(fg1[gt1].mean()) if gt1.any() else np.nan,
        "fg_mean_gtneg_state1": float(fg1[~gt1].mean()) if (~gt1).any() else np.nan,
        "frac_fg_gt0p5_state2": float((pred[s2] > 0.5).mean()) if s2.any() else np.nan,
        "mean_fg_state2": float(pred[s2].mean()) if s2.any() else np.nan,
        "ap": np.nan,
        "auc": np.nan,
    }
    if gt1.any() and (~gt1).any():
        gt_sub, fg_sub = _subsample(gt1, fg1, MAX_POINTS_AP)
        scores["ap"] = float(average_precision_score(gt_sub, fg_sub))
        scores["auc"] = float(roc_auc_score(gt_sub, fg_sub))
    return scores


def get_regions(state, band_nm, voxel_size):
    """Get the regions for the AP: the annotated mitochondria, and a membrane shell and its complement.

    The regions are returned as mitochondria states that are 1 only inside of the region, so that
    evaluate_prediction restricts to them.
    """
    regions = [("all", state)]
    if not band_nm:
        return regions
    mito = state == 1
    # The distance to the nearest voxel outside of the annotated mitochondria, in nanometer.
    if mito.any():
        distance = distance_transform_edt(mito, sampling=voxel_size).astype(np.float32)
    else:
        distance = np.zeros(mito.shape, dtype=np.float32)
    for thickness in band_nm:
        band = mito & (distance <= float(thickness))
        regions.append((f"band{thickness:g}", band.astype(np.uint8)))
        regions.append((f"core{thickness:g}", (mito & ~band).astype(np.uint8)))
    return regions


def evaluate_file(label_path, seg_path, segment_key, anno_key, band_nm, voxel_size):
    print("Evaluate", seg_path, "against", label_path)
    with open_file(label_path, "r") as f:
        labels = f[anno_key][:]
        state = f["raw_mitos_combined"][1]
    with open_file(seg_path, "r") as f:
        seg = f[segment_key][:]
        pred = f["pred/foreground"][:].astype(np.float32) if "pred/foreground" in f else None

    tomogram = os.path.basename(label_path)
    seg_scores = {"tomogram": tomogram, **evaluate_segmentation(labels, seg, state)}
    ap_scores = []
    if pred is not None:
        for region, region_state in get_regions(state, band_nm, voxel_size):
            ap_scores.append({"tomogram": tomogram, "region": region,
                              **evaluate_prediction(labels, pred, region_state)})
    return seg_scores, ap_scores


def summarize(seg_results, ap_results):
    """Add the average over the tomograms: the mean of the scores and the sum of the voxel counts."""
    counts = ["tp", "fp", "fn", "pred_fg", "gt_fg", "eval_voxels"]
    average = {"tomogram": "all"}
    for column in seg_results.columns.drop("tomogram"):
        average[column] = seg_results[column].sum() if column in counts else np.nanmean(seg_results[column])
    seg_results = pd.concat([seg_results, pd.DataFrame([average])], ignore_index=True)

    if len(ap_results) > 0:
        averages = []
        for region, results in ap_results.groupby("region", sort=False):
            averages.append({"tomogram": "all", "region": region,
                             **results.drop(columns=["tomogram", "region"]).mean().to_dict()})
        ap_results = pd.concat([ap_results, pd.DataFrame(averages)], ignore_index=True)
    return seg_results, ap_results


def find_label_file(seg_path, label_paths):
    name = os.path.basename(seg_path)
    # Prefer the file with the same name, so that a name contained in another one cannot match it.
    for label_path in label_paths:
        if os.path.basename(label_path) == name:
            return label_path
    for label_path in label_paths:
        if os.path.splitext(name)[0] in os.path.basename(label_path):
            return label_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate the cristae segmentation on the test set.")
    parser.add_argument("-sp", "--segmentation_path", required=True,
                        help="The folder with the segmentations written by segment_test_set.py, or a single file.")
    parser.add_argument("-gp", "--groundtruth_path", default=None,
                        help="The folder or file with the annotations. By default the test set of the split file.")
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-sk", "--segmentation_key", default="seg")
    parser.add_argument("-gk", "--groundtruth_key", default="labels/cristae")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--band_nm", type=float, nargs="*", default=[8.0, 12.0],
                        help="The thicknesses of the membrane shells for the AP, in nanometer.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=CRISTAE_VOXEL_SIZE,
                        help="The voxel size (z, y, x) in nanometer, for the membrane shells.")
    args = parser.parse_args()

    if args.groundtruth_path is None:
        label_paths = get_cristae_test_paths(SPLIT_FILE, TRAIN_ROOTS)
    elif os.path.isdir(args.groundtruth_path):
        label_paths = sorted(glob(os.path.join(args.groundtruth_path, "**", "*.h5"), recursive=True))
    else:
        label_paths = [args.groundtruth_path]
    if os.path.isdir(args.segmentation_path):
        seg_paths = sorted(glob(os.path.join(args.segmentation_path, "*.h5")))
    else:
        seg_paths = [args.segmentation_path]

    seg_results, ap_results = [], []
    for seg_path in seg_paths:
        if any(name in os.path.basename(seg_path) for name in EXCLUDE):
            print("Skipping", seg_path, "because its raw data or annotations are unreliable")
            continue
        label_path = find_label_file(seg_path, label_paths)
        if label_path is None:
            print("Skipping", seg_path, "because there is no label file for it")
            continue
        seg_scores, ap_scores = evaluate_file(
            label_path, seg_path, args.segmentation_key, args.groundtruth_key, args.band_nm, tuple(args.voxel_size)
        )
        seg_results.append(seg_scores)
        ap_results.extend(ap_scores)
    seg_results, ap_results = summarize(pd.DataFrame(seg_results), pd.DataFrame(ap_results))

    os.makedirs(args.output_folder, exist_ok=True)
    seg_results.to_csv(os.path.join(args.output_folder, f"evaluation_{args.model_name}.csv"), index=False)
    print(seg_results[["tomogram", "dice", "precision", "recall"]].to_markdown(index=False))
    if len(ap_results) > 0:
        ap_results.to_csv(os.path.join(args.output_folder, f"evaluation_{args.model_name}_ap.csv"), index=False)
        averages = ap_results[ap_results.tomogram == "all"]
        print(averages[["region", "ap", "auc"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
