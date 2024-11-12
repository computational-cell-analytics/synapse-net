import os
from glob import glob

import h5py
from tqdm import tqdm


INPUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/04Dataset_for_vesicle_eval/model_segmentation"  # noqa
OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/04_full_reconstruction"


def assort_az_and_vesicles(in_path, out_path):
    if os.path.exists(out_path):
        return

    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]
        vesicles = f["/vesicles/segment_from_combined_vesicles"][:]
        az = f["/AZ/segment_from_AZmodel_v3"][:]

    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    with h5py.File(out_path, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/vesicles", data=vesicles, compression="gzip")
        f.create_dataset("labels/active_zone", data=az, compression="gzip")


def main():
    paths = sorted(glob(os.path.join(INPUT_ROOT, "**/*.h5"), recursive=True))
    for path in tqdm(paths):
        fname = os.path.relpath(path, INPUT_ROOT)
        out_path = os.path.join(OUTPUT_ROOT, fname)
        assort_az_and_vesicles(path, out_path)


if __name__ == "__main__":
    main()
