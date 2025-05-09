import os

import h5py
import pandas as pd
from elf.evaluation import dice_score

from scipy.ndimage import binary_dilation, binary_closing
from tqdm import tqdm


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


main()
