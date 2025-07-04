import os
from glob import glob

import h5py
import pandas as pd
from elf.evaluation.matching import matching
from elf.evaluation.dice import symmetric_best_dice_score
from tqdm import tqdm


INPUT_FOLDER = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Automatische_Segmentierung_Dataset_Validierung"  # noqa
OUTPUT_FOLDER = "./predictions/endbulb"


def evaluate_dataset(ds_name="endbulb", force=False):
    result_folder = "./results/endbulb"
    os.makedirs(result_folder, exist_ok=True)
    result_path = os.path.join(result_folder, f"{ds_name}.csv")
    if os.path.exists(result_path) and not force:
        results = pd.read_csv(result_path)
        return results

    print("Evaluating ds", ds_name)
    input_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.h5")))
    pred_files = sorted(glob(os.path.join(OUTPUT_FOLDER, "*.h5")))

    results = {
        "dataset": [],
        "file": [],
        "precision": [],
        "recall": [],
        "f1-score": [],
        "sbd-score": [],
    }
    for inf, predf in tqdm(zip(input_files, pred_files), total=len(input_files), desc="Run evaluation"):
        fname = os.path.basename(inf)

        with h5py.File(inf, "r") as f:
            gt = f["/labels/vesicles"][:]
        with h5py.File(predf, "r") as f:
            seg = f["/prediction/vesicles/cryovesnet"][:]
        assert gt.shape == seg.shape

        scores = matching(seg, gt)
        sbd_score = symmetric_best_dice_score(seg, gt)

        results["dataset"].append(ds_name)
        results["file"].append(fname)
        results["precision"].append(scores["precision"])
        results["recall"].append(scores["recall"])
        results["f1-score"].append(scores["f1"])
        results["sbd-score"].append(sbd_score)

    results = pd.DataFrame(results)
    results.to_csv(result_path, index=False)
    return results


def main():
    force = False
    result = evaluate_dataset(force=force)
    print(result)
    print()
    print("F1-Score")
    print(result["f1-score"].mean(), "+-", result["f1-score"].std())
    print("SBD-Score")
    print(result["sbd-score"].mean(), "+-", result["sbd-score"].std())


if __name__ == "__main__":
    main()
