import os
from glob import glob

import h5py
from synaptic_reconstruction.inference.mitochondria import segment_mitochondria
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/04_full_reconstruction"  # noqa
MODEL_PATH = "/scratch-grete/projects/nim00007/models/exports_for_cooper/mito_model_s2.pt"  # noqa


def run_seg(path):
    with h5py.File(path, "r") as f:
        if "labels/mitochondria" in f:
            return
        raw = f["raw"][:]

    scale = (0.5, 0.5, 0.5)
    seg = segment_mitochondria(raw, model_path=MODEL_PATH, scale=scale, verbose=False)
    with h5py.File(path, "a") as f:
        f.create_dataset("labels/mitochondria", data=seg, compression="gzip")


def main():
    paths = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    for path in tqdm(paths):
        run_seg(path)


main()
