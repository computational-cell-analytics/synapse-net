import os
from glob import glob

import h5py
from tqdm import tqdm

from synaptic_reconstruction.inference.util import _Scaler
from synaptic_reconstruction.inference.compartments import segment_compartments

INPUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/04Dataset_for_vesicle_eval"  # noqa
# MODEL_PATH = "/mnt/lustre-emmy-hdd/projects/nim00007/compartment_models/compartment_model_3d.pt"
MODEL_PATH = "/user/pape41/u12086/Work/my_projects/synaptic-reconstruction/scripts/cooper/training/checkpoints/compartment_model_3d/v2"  # noqa
OUTPUT = "/mnt/lustre-emmy-hdd/projects/nim00007/compartment_predictions"


def label_transform_3d():
    pass


def segment_volume(input_path, model_path):
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]

    scale = (0.25, 0.25, 0.25)
    scaler = _Scaler(scale, verbose=False)
    raw = scaler.scale_input(raw)

    n_slices_exclude = 2
    seg, pred = segment_compartments(
        raw, model_path, verbose=False, n_slices_exclude=n_slices_exclude, return_predictions=True
    )
    # raw, seg = raw[n_slices_exclude:-n_slices_exclude], seg[n_slices_exclude:-n_slices_exclude]

    return raw, seg, pred


def main():
    inputs = sorted(glob(os.path.join(INPUT_ROOT, "**/*.h5"), recursive=True))
    inputs = [inp for inp in inputs if "cropped_for_2D" not in inp]

    for input_path in tqdm(inputs, desc="Run prediction for 04."):
        ds_name, fname = os.path.split(input_path)
        ds_name = os.path.split(ds_name)[1]
        output_folder = os.path.join(OUTPUT, "segmentation", ds_name)
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            continue

        pred_folder = os.path.join(OUTPUT, "prediction", ds_name)
        os.makedirs(pred_folder, exist_ok=True)
        pred_path = os.path.join(pred_folder, fname)

        raw, seg, pred = segment_volume(input_path, MODEL_PATH)
        os.makedirs(output_folder, exist_ok=True)
        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/compartments", data=seg, compression="gzip")
        with h5py.File(pred_path, "a") as f:
            f.create_dataset("prediction", data=pred, compression="gzip")


if __name__ == "__main__":
    main()
