import os
from glob import glob

import h5py
from synaptic_reconstruction.inference.compartments import segment_compartments
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/04_full_reconstruction"  # noqa
MODEL_PATH = "/user/pape41/u12086/Work/my_projects/synaptic-reconstruction/scripts/cooper/training/checkpoints/compartment_model_3d/v2"  # noqa


def label_transform_3d():
    pass


def run_seg(path):
    with h5py.File(path, "r") as f:
        if "labels/compartments" in f:
            return
        raw = f["raw"][:]

    scale = (0.25, 0.25, 0.25)
    seg = segment_compartments(raw, model_path=MODEL_PATH, scale=scale, verbose=False)
    with h5py.File(path, "a") as f:
        f.create_dataset("labels/compartments", data=seg, compression="gzip")


def main():
    paths = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    for path in tqdm(paths):
        run_seg(path)


main()
