import os
import h5py
import numpy as np
import pandas as pd

from synapse_net.inference.inference import get_model
from synapse_net.inference.compartments import segment_compartments
from skimage.segmentation import find_boundaries

from elf.evaluation.matching import matching

from train_compartments import get_paths_3d
from sklearn.model_selection import train_test_split


def run_prediction(paths):
    output_folder = "./compartment_eval"
    os.makedirs(output_folder, exist_ok=True)

    model = get_model("compartments")
    for path in paths:
        with h5py.File(path, "r") as f:
            input_vol = f["raw"][:]
        seg, pred = segment_compartments(input_vol, model=model, return_predictions=True)
        fname = os.path.basename(path)
        out = os.path.join(output_folder, fname)
        with h5py.File(out, "a") as f:
            f.create_dataset("seg", data=seg, compression="gzip")
            f.create_dataset("pred", data=pred, compression="gzip")


def binary_recall(gt, pred):
    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    return float(tp) / (tp + fn) if (tp + fn) else 0.0


def run_evaluation(paths):
    output_folder = "./compartment_eval"

    results = {
        "name": [],
        "recall-pred": [],
        "recall-seg": [],
    }

    for path in paths:
        with h5py.File(path, "r") as f:
            labels = f["labels/compartments"][:]
        boundary_labels = find_boundaries(labels).astype("bool")

        fname = os.path.basename(path)
        out = os.path.join(output_folder, fname)
        with h5py.File(out, "a") as f:
            seg, pred = f["seg"][:], f["pred"][:]

        recall_pred = binary_recall(boundary_labels, pred > 0.5)
        recall_seg = matching(seg, labels)["recall"]

        results["name"].append(fname)
        results["recall-pred"].append(recall_pred)
        results["recall-seg"].append(recall_seg)

    results = pd.DataFrame(results)
    print(results)
    print(results[["recall-pred", "recall-seg"]].mean())


def check_predictions(paths):
    import napari
    output_folder = "./compartment_eval"

    for path in paths:
        with h5py.File(path, "r") as f:
            raw = f["raw"][:]
            labels = f["labels/compartments"][:]
        boundary_labels = find_boundaries(labels)

        fname = os.path.basename(path)
        out = os.path.join(output_folder, fname)
        with h5py.File(out, "a") as f:
            seg, pred = f["seg"][:], f["pred"][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(pred)
        v.add_labels(labels)
        v.add_labels(boundary_labels)
        v.add_labels(seg)
        napari.run()


def main():
    paths = get_paths_3d()
    _, val_paths = train_test_split(paths, test_size=0.10, random_state=42)

    # run_prediction(val_paths)
    run_evaluation(val_paths)
    # check_predictions(val_paths)


if __name__ == "__main__":
    main()
