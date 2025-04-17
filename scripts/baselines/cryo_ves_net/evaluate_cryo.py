import os
from glob import glob

import h5py
import pandas as pd
from elf.evaluation.matching import matching
from elf.evaluation.dice import symmetric_best_dice_score


INPUT_FOLDER = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/vesicle_gt/v3"  # noqa
OUTPUT_FOLDER = "./predictions/cryo"


def evaluate_dataset(ds_name="cryo", force=False):
    result_folder = "./results/cryo"
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
    for inf, predf in zip(input_files, pred_files):
        fname = os.path.basename(inf)

        with h5py.File(inf, "r") as f:
            gt = f["/labels/vesicles"][:]
        with h5py.File(predf, "r") as f:
            seg = f["/prediction/vesicles/cryovesnet"][:]
        assert gt.shape == seg.shape

        scores = matching(seg, gt)
        sbd_score = symmetric_best_dice_score(seg, gt)
        scores["sbd"] = sbd_score

        results["dataset"].append(ds_name)
        results["file"].append(fname)
        results["precision"].append(scores["precision"])
        results["recall"].append(scores["recall"])
        results["f1-score"].append(scores["f1"])
        results["sbd-score"].append(scores["sbd"])

    results = pd.DataFrame(results)
    results.to_csv(result_path, index=False)
    return results


def main():
    force = False
    result = evaluate_dataset(force=force)
    print(result)
    print("F1-Score")
    print(result["f1-score"].mean())
    print("SBD-Score")
    print(result["sbd-score"].mean())


if __name__ == "__main__":
    main()
