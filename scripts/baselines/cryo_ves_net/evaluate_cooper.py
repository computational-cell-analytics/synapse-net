import json
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from elf.evaluation.matching import matching
from tqdm import tqdm

INPUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets"  # noqa
INPUT_04 = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/04Dataset_for_vesicle_eval"  # noqa
OUTPUT_ROOT = "./predictions/cooper"  # noqa

DATASETS = [
    "01_hoi_maus_2020_incomplete",
    "02_hcc_nanogold",
    "03_hog_cs1sy7",
    "04",
    "05_stem750_sv_training",
    "07_hoi_s1sy7_tem250_ihgp",
    "10_tem_single_release",
    "11_tem_multiple_release",
    "12_chemical_fix_cryopreparation"
]


def evaluate_dataset(ds_name):
    result_folder = "./results/cooper"
    os.makedirs(result_folder, exist_ok=True)
    result_path = os.path.join(result_folder, f"{ds_name}.csv")
    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
        return results

    print("Evaluating ds", ds_name)
    if ds_name == "04":
        input_files = sorted(glob(os.path.join(INPUT_04, "**/*.h5"), recursive=True))
        seg_key = "labels/vesicles"
        mask_key = "labels/compartment"
    else:
        input_files = sorted(glob(os.path.join(INPUT_ROOT, ds_name, "**/*.h5"), recursive=True))
        seg_key = "/labels/vesicles/combined_vesicles"
        mask_key = None

    pred_files = sorted(glob(os.path.join(OUTPUT_ROOT, ds_name, "**/*.h5"), recursive=True))
    assert len(input_files) == len(pred_files), f"{len(input_files)}, {len(pred_files)}"

    results = {
        "dataset": [],
        "file": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
    }
    for inf, predf in tqdm(zip(input_files, pred_files), total=len(input_files), desc=f"Evaluate {ds_name}"):
        fname = os.path.basename(inf)
        sub_res_path = os.path.join(result_folder, f"{ds_name}_{fname}.json")

        if os.path.exists(sub_res_path):
            print("Loading scores from", sub_res_path)
            with open(sub_res_path, "r") as f:
                scores = json.load(f)

        else:
            try:
                with h5py.File(predf, "r") as f:
                    seg = f["/prediction/vesicles/cryovesnet"][:]
            except Exception:
                print("Skipping", predf)
                continue

            with h5py.File(inf, "r") as f:
                gt = f[seg_key][:]
                if mask_key is None:
                    mask = None
                else:
                    mask = f[mask_key][:]

            assert gt.shape == seg.shape

            if mask is not None:
                bb = np.where(mask != 0)
                bb = tuple(slice(
                    int(b.min()), int(b.max()) + 1
                ) for b in bb)
                seg, gt, mask = seg[bb], gt[bb], mask[bb]
                seg[mask == 0] = 0
                gt[mask == 0] = 0

            scores = matching(seg, gt)

            with open(sub_res_path, "w") as f:
                json.dump(scores, f)

        results["dataset"].append(ds_name)
        results["file"].append(fname)
        results["precision"].append(scores["precision"])
        results["recall"].append(scores["recall"])
        results["f1-score"].append(scores["f1"])

    results = pd.DataFrame(results)
    results.to_csv(result_path, index=False)
    return results


def main():
    all_results = {}
    for ds in DATASETS:
        result = evaluate_dataset(ds)
        all_results[ds] = result

    groups = {
        "Chemical Fixation": ["12_chemical_fix_cryopreparation"],
        "Single-Axis-TEM": ["01_hoi_maus_2020_incomplete"],
        "Dual-Axis-TEM": ["02_hcc_nanogold",
                          "03_hog_cs1sy7",
                          "07_hoi_s1sy7_tem250_ihgp",
                          "10_tem_single_release",
                          "11_tem_multiple_release"],
        "STEM": ["04", "05_stem750_sv_training"],
        "Overall": DATASETS,
    }

    for name, datasets in groups.items():
        f1_scores = []

        for ds in datasets:
            this_f1_scores = all_results[ds]["f1-score"].values.tolist()
            f1_scores.extend(this_f1_scores)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(name, ":", mean_f1, "+-", std_f1)


if __name__ == "__main__":
    main()
