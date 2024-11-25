import os
from glob import glob

import h5py
import pandas as pd

from elf.evaluation.dice import dice_score
from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.postprocessing.ribbon import segment_ribbon
from synaptic_reconstruction.inference.postprocessing.presynaptic_density import segment_presynaptic_density
from torch_em.util import load_model
from tqdm import tqdm

from train_structure_segmentation import get_train_val_test_split

# ROOT = "/home/pape/Work/data/synaptic_reconstruction/moser"
ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser"
MODEL_PATH = "/mnt/lustre-emmy-hdd/projects/nim00007/models/synaptic-reconstruction/vesicle-DA-inner_ear-v2"
OUTPUT_ROOT = "./predictions"


def run_vesicle_segmentation(input_paths, model_path, name, is_nested=False):
    output_root = os.path.join(OUTPUT_ROOT, name)
    model = None

    for path in input_paths:
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if "vesicles" in f:
                    continue

        if model is None:
            model = load_model(model_path)

        with h5py.File(path, "r") as f:
            tomogram = f["raw"][:]

        seg = segment_vesicles(input_volume=tomogram, model=model)
        with h5py.File(output_path, "a") as f:
            f.create_dataset("vesicles", data=seg, compression="gzip")


def postprocess_structures(paths, name, prefix=None, is_nested=False):
    output_root = os.path.join(OUTPUT_ROOT, name)

    for path in tqdm(paths):
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root
        output_path = os.path.join(output_folder, fname)

        with h5py.File(output_path, "r") as f:
            if prefix is None and "segmentation" in f:
                continue
            elif prefix is not None and f"{prefix}/segmentation" in f:
                continue

            vesicles = f["vesicles"][:]
            if prefix is None:
                ribbon_pred = f["ribbon"][:]
                presyn_pred = f["PD"][:]
            else:
                ribbon_pred = f[f"{prefix}/ribbon"][:]
                presyn_pred = f[f"{prefix}/PD"][:]

        # import napari
        # v = napari.Viewer()
        # v.add_image(ribbon_pred)
        # v.add_image(presyn_pred)
        # v.add_labels(vesicles)
        # napari.run()

        ribbon = segment_ribbon(ribbon_pred, vesicles, n_slices_exclude=15, n_ribbons=1)
        presyn = segment_presynaptic_density(presyn_pred, ribbon, n_slices_exclude=15)

        with h5py.File(output_path, "a") as f:
            if prefix is None:
                f.create_dataset("segmentation/ribbon", data=ribbon, compression="gzip")
                f.create_dataset("segmentation/PD", data=presyn, compression="gzip")
            else:
                f.create_dataset(f"{prefix}/segmentation/ribbon", data=ribbon, compression="gzip")
                f.create_dataset(f"{prefix}/segmentation/PD", data=presyn, compression="gzip")


def visualize(input_paths, name, is_nested=False, label_names=None, prefixes=None):
    import napari

    structure_names = ["ribbon", "PD"]
    if label_names is None:
        label_names = structure_names

    output_root = os.path.join(OUTPUT_ROOT, name)
    for path in input_paths:
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root
        output_path = os.path.join(output_folder, fname)

        labels = {}
        with h5py.File(path, "r") as f:
            raw = f["raw"][:]
            for name, sname in zip(label_names, structure_names):
                labels[name] = f[f"labels/{name}"][:]

        predictions = {}
        with h5py.File(output_path, "r") as f:
            if prefixes is None:
                for name in structure_names:
                    predictions[name] = f[f"segmentation/{name}"][:]
            else:
                for prefix in prefixes:
                    for name in structure_names:
                        predictions[f"{prefix}/{name}"] = f[f"{prefix}/segmentation/{name}"][:]

        v = napari.Viewer()
        v.add_image(raw)
        for name, seg in labels.items():
            v.add_labels(seg, name=f"labels/{name}", visible=False)
        for name, seg in predictions.items():
            if name == "ribbon":
                cmap = {1: "orange"}
            else:
                cmap = {1: "green"}
            v.add_labels(seg, name=name, colormap=cmap)
        v.title = fname
        napari.run()


def evaluate(input_paths, name, is_nested=False, prefix=None, save_path=None, label_names=None):
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    structure_names = ["ribbon", "PD"]
    if label_names is None:
        label_names = structure_names
    output_root = os.path.join(OUTPUT_ROOT, name)

    results = {
        "method": [],
        "file_name": [],
    }
    results.update({nn: [] for nn in structure_names})
    for path in tqdm(input_paths, desc="Run evaluation"):
        root, fname = os.path.split(path)
        if is_nested:
            folder_name = os.path.split(root)[1]
            output_folder = os.path.join(output_root, folder_name)
        else:
            output_folder = output_root
        output_path = os.path.join(output_folder, fname)

        results["method"].append("Src" if prefix is None else prefix)
        results["file_name"].append(f"{folder_name}/{fname}" if is_nested else fname)

        with h5py.File(path, "r") as f_in, h5py.File(output_path, "r") as f_out:
            for sname, label_name in zip(structure_names, label_names):
                gt = f_in[f"labels/{label_name}"][:]
                pred = f_out[f"segmentation/{sname}" if prefix is None else f"{prefix}/segmentation/{sname}"][:]
                score = dice_score(pred, gt)
                results[sname].append(score)

    results = pd.DataFrame(results)
    if save_path is not None:
        results.to_csv(save_path, index=False)
    return results


def segment_train_domain():
    _, _, paths = get_train_val_test_split(os.path.join(ROOT, "inner_ear_data"))
    print("Run evaluation on", len(paths), "tomos")
    name = "train_domain"
    run_vesicle_segmentation(paths, MODEL_PATH, name, is_nested=True)
    postprocess_structures(paths, name, is_nested=True)
    # visualize(paths, name, is_nested=True)
    results = evaluate(paths, name, is_nested=True, save_path="./results/train_domain_postprocessed_v2.csv")
    print(results)
    print("Ribbon segmentation:", results["ribbon"].mean(), "+-", results["ribbon"].std())
    print("PD segmentation:", results["PD"].mean(), "+-", results["PD"].std())


def segment_vesicle_pools():
    paths = sorted(glob(os.path.join(ROOT, "other_tomograms/01_vesicle_pools", "*.h5")))
    run_vesicle_segmentation(paths, MODEL_PATH, "vesicle_pools")

    name = "vesicle_pools"
    prefixes = ("Src", "Adapted")
    label_names = ["ribbons", "presynapse", "membrane"]

    for prefix in prefixes:
        postprocess_structures(paths, name, prefix=prefix, is_nested=False)

        save_path = f"./results/{name}_{prefix}.csv"
        results = evaluate(paths, name, prefix=prefix, save_path=save_path, label_names=label_names)
        print("Results for", name, prefix, ":")
        print(results)

    # visualize(paths, name, label_names=label_names, prefixes=prefixes)


def segment_rat():
    paths = sorted(glob(os.path.join(ROOT, "other_tomograms/03_ratten_tomos", "*.h5")))
    run_vesicle_segmentation(paths, MODEL_PATH, "rat")

    name = "rat"
    prefixes = ("Src", "Adapted")
    label_names = ["ribbons", "presynapse", "membrane"]

    for prefix in prefixes:
        postprocess_structures(paths, name, prefix=prefix, is_nested=False)

        save_path = f"./results/{name}_{prefix}.csv"
        results = evaluate(paths, name, prefix=prefix, save_path=save_path, label_names=label_names)
        print("Results for", name, prefix, ":")
        print(results)

    # visualize(paths, name, label_names=label_names, prefixes=prefixes)


def main():
    segment_train_domain()
    # segment_vesicle_pools()
    # segment_rat()


if __name__ == "__main__":
    main()
