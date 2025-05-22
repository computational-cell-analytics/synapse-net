import os
import argparse

import h5py
import pandas as pd
import numpy as np
from elf.evaluation import dice_score

from scipy.ndimage import binary_dilation, binary_closing
from tqdm import tqdm
from elf.evaluation import matching


def _expand_AZ(az):
    return binary_closing(
        binary_dilation(az, iterations=3), iterations=3
    )


def eval_az(seg_path, gt_path, seg_key, gt_key):
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][:]
    with h5py.File(gt_path, "r") as f:
        gt = f[gt_key][:]
    assert seg.shape == gt.shape

    seg = _expand_AZ(seg)
    gt = _expand_AZ(gt)
    score = dice_score(seg, gt)

    # import napari
    # v = napari.Viewer()
    # v.add_labels(seg)
    # v.add_labels(gt)
    # v.title = f"Dice = {score}, {seg_path}"
    # napari.run()

    return score


def main():
    res_path = "./az_eval.xlsx"
    if not os.path.exists(res_path):
        seg_root = "AZ_segmentation/postprocessed_AZ"
        gt_root = "postprocessed_AZ"

        # Removed WT_Unt_SC_05646_D4_02_DIV16_mtk_02.h5 from the eval set because of contrast issues
        test_tomograms = {
            "01": [
                "WT_MF_DIV28_01_MS_09204_F1.h5", "WT_MF_DIV14_01_MS_B2_09175_CA3.h5", "M13_CTRL_22723_O2_05_DIV29_5.2.h5", "WT_Unt_SC_09175_D4_05_DIV14_mtk_05.h5",  # noqa
                "20190805_09002_B4_SC_11_SP.h5", "20190807_23032_D4_SC_01_SP.h5", "M13_DKO_22723_A1_03_DIV29_03_MS.h5", "WT_MF_DIV28_05_MS_09204_F1.h5", "M13_CTRL_09201_S2_06_DIV31_06_MS.h5", # noqa
                "WT_MF_DIV28_1.2_MS_09002_B1.h5", "WT_Unt_SC_09175_C4_04_DIV15_mtk_04.h5",   "M13_DKO_22723_A4_10_DIV29_10_MS.h5",  "WT_MF_DIV14_3.2_MS_D2_09175_CA3.h5",  # noqa
                   "20190805_09002_B4_SC_10_SP.h5", "M13_CTRL_09201_S2_02_DIV31_02_MS.h5", "WT_MF_DIV14_04_MS_E1_09175_CA3.h5", "WT_MF_DIV28_10_MS_09002_B3.h5",  "M13_DKO_22723_A4_08_DIV29_08_MS.h5",  "WT_MF_DIV28_04_MS_09204_M1.h5",   "WT_MF_DIV28_03_MS_09204_F1.h5",   "M13_DKO_22723_A1_05_DIV29_05_MS.h5",  # noqa
                "WT_Unt_SC_09175_C4_06_DIV15_mtk_06.h5",  "WT_MF_DIV28_09_MS_09002_B3.h5", "20190524_09204_F4_SC_07_SP.h5",
                   "WT_MF_DIV14_02_MS_C2_09175_CA3.h5",    "M13_DKO_23037_K1_01_DIV29_01_MS.h5",  "WT_Unt_SC_09175_E2_01_DIV14_mtk_01.h5", "20190807_23032_D4_SC_05_SP.h5",   "WT_MF_DIV14_01_MS_E2_09175_CA3.h5",   "WT_MF_DIV14_03_MS_B2_09175_CA3.h5",   "M13_DKO_09201_O1_01_DIV31_01_MS.h5",  "M13_DKO_09201_U1_04_DIV31_04_MS.h5",  # noqa
                "WT_MF_DIV14_04_MS_E2_09175_CA3_2.h5",   "WT_Unt_SC_09175_D5_01_DIV14_mtk_01.h5",
                "M13_CTRL_22723_O2_05_DIV29_05_MS_.h5",  "WT_MF_DIV14_02_MS_B2_09175_CA3.h5", "WT_MF_DIV14_01.2_MS_D1_09175_CA3.h5",  # noqa
            ],
            "12": ["20180305_09_MS.h5", "20180305_04_MS.h5", "20180305_08_MS.h5",
                   "20171113_04_MS.h5", "20171006_05_MS.h5", "20180305_01_MS.h5"],
        }

        scores = {
            "Dataset": [],
            "Tomogram": [],
            "Dice": []
        }
        for ds, test_tomos in test_tomograms.items():
            ds_name = "01_hoi_maus_2020_incomplete" if ds == "01" else "12_chemical_fix_cryopreparation"
            for tomo in tqdm(test_tomos):
                seg_path = os.path.join(seg_root, ds_name, tomo)
                gt_path = os.path.join(gt_root, ds_name, tomo)
                score = eval_az(seg_path, gt_path, seg_key="AZ/thin_az", gt_key="labels_pp/filtered_az")

                scores["Dataset"].append(ds_name)
                scores["Tomogram"].append(tomo)
                scores["Dice"].append(score)

        scores = pd.DataFrame(scores)
        scores.to_excel(res_path, index=False)

    else:
        scores = pd.read_excel(res_path)

    print("Evaluation for the datasets:")
    for ds in pd.unique(scores.Dataset):
        print(ds)
        ds_scores = scores[scores.Dataset == ds]["Dice"]
        print(ds_scores.mean(), "+-", ds_scores.std())

    print("Total:")
    print(scores["Dice"].mean(), "+-", scores["Dice"].std())

def get_bounding_box(mask, halo=2):
    """ Get bounding box coordinates around a mask with a halo. """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None  # No labels found

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    min_coords = np.maximum(min_coords - halo, 0)
    max_coords = np.minimum(max_coords + halo, mask.shape)

    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return slices


def evaluate(labels, seg):
    assert labels.shape == seg.shape
    stats = matching(seg, labels)
    return [stats["f1"], stats["precision"], stats["recall"]]


def evaluate_file(labels_path, seg_path, segment_key, anno_key, mask_key=None):
    print(f"Evaluating: {os.path.basename(labels_path)}")

    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)

    with h5py.File(labels_path) as label_file:
        labels = label_file["labels"]
        gt = labels[anno_key][:]
        if mask_key is not None:
            mask = labels[mask_key][:]

    with h5py.File(seg_path) as seg_file:
        az = seg_file["AZ"][segment_key][:]

    if mask_key is not None:
        gt[mask == 0] = 0
        az[mask == 0] = 0

    # Optionally crop to bounding box
    use_bb = False
    if use_bb:
        bb_slices = get_bounding_box(gt, halo=2)
        gt = gt[bb_slices]
        az = az[bb_slices]

    scores = evaluate(gt, az)
    return pd.DataFrame([[ds_name, tomo] + scores],
                        columns=["dataset", "tomogram", "f1-score", "precision", "recall"])


def evaluate_folder(labels_path, seg_path, model_name, segment_key, anno_key, mask_key=None):
    print(f"\nEvaluating folder: {seg_path}")
    label_files = os.listdir(labels_path)
    seg_files = os.listdir(seg_path)

    all_results = []

    for seg_file in seg_files:
        if seg_file in label_files:
            label_fp = os.path.join(labels_path, seg_file)
            seg_fp = os.path.join(seg_path, seg_file)

            res = evaluate_file(label_fp, seg_fp, segment_key, anno_key, mask_key)
            all_results.append(res)

    if not all_results:
        print("No matched files found for evaluation.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Per-folder metrics
    folder_metrics = results_df[["f1-score", "precision", "recall"]].mean().to_dict()
    folder_name = os.path.basename(seg_path)
    print(f"\n Folder-Level Metrics for '{folder_name}':")
    print(f"Precision: {folder_metrics['precision']:.4f}")
    print(f"Recall:    {folder_metrics['recall']:.4f}")
    print(f"F1 Score:  {folder_metrics['f1-score']:.4f}\n")

    # Save results
    result_dir = "/user/muth9/u12095/synapse-net/scripts/cooper/evaluation_results"
    os.makedirs(result_dir, exist_ok=True)

    # Per-file CSV
    file_results_path = os.path.join(result_dir, f"evaluation_{model_name}_per_file.csv")
    if os.path.exists(file_results_path):
        existing_df = pd.read_csv(file_results_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    results_df.to_csv(file_results_path, index=False)
    print(f"Per-file results saved to {file_results_path}")

    # Append to per-folder summary
    folder_summary_path = os.path.join(result_dir, f"evaluation_{model_name}_per_folder.csv")
    with open(folder_summary_path, "a") as f:
        f.write(f"{folder_name},{folder_metrics['f1-score']:.4f},{folder_metrics['precision']:.4f},{folder_metrics['recall']:.4f}\n")
    print(f"Folder summary appended to {folder_summary_path}")


def main_f1():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True)
    parser.add_argument("-seg", "--seg_path", required=True)
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-sk", "--segment_key", required=True)
    parser.add_argument("-ak", "--anno_key", required=True)
    parser.add_argument("-m", "--mask_key")

    args = parser.parse_args()

    if os.path.isdir(args.seg_path):
        evaluate_folder(args.labels_path, args.seg_path, args.model_name, args.segment_key, args.anno_key, args.mask_key)
    else:
        print("Please pass a folder to get folder-level metrics.")


if __name__ == "__main__":
    #main() #for dice score
    main_f1()