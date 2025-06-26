import argparse
import os
from glob import glob

import h5py
import napari
import numpy as np
from scipy.ndimage import binary_closing
from common import ALL_NAMES, get_file_names, get_split_folder, get_paths


SKIP_MERGE = [
    "36859_J1_66K_TS_CA3_PS_26_rec_2Kb1dawbp_crop.h5",
    "36859_J1_66K_TS_CA3_PS_23_rec_2Kb1dawbp_crop.h5",
    "36859_J1_66K_TS_CA3_PS_23_rec_2Kb1dawbp_crop.h5",
    "36859_J1_STEM750_66K_SP_17_rec_2kb1dawbp_crop.h5",
]


# STEM CROPPED IS OFTEN TOO SMALL!
def merge_az(name, version, check):
    split_folder = get_split_folder(version)
   
    if name == "stem_cropped":
        file_paths = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped", "*.h5"))
        file_names = [os.path.basename(path) for path in file_paths]
    else:
        file_names = get_file_names(name, split_folder, split_names=["train", "val", "test"])
    seg_paths, gt_paths = get_paths(name, file_names)

    for seg_path, gt_path in zip(seg_paths, gt_paths):

        with h5py.File(gt_path, "r") as f:
            #if not check and ("labels/az_merged" in f):
            if f"labels/az_merged_v{version}" in f :
                continue
            raw = f["raw"][:]
            gt = f["labels/az"][:]
            gt_thin = f["labels/az_thin"][:]

        with h5py.File(seg_path) as f:
            seg_key = f"predictions/az/v{version}"
            pred = f[seg_key][:]

        fname = os.path.basename(seg_path)
        if fname in SKIP_MERGE:
            az_merged = gt
        else:
            threshold = 0.4
            gt_ = np.logical_or(binary_closing(gt, iterations=4), gt)
            seg = pred > threshold
            az_merged = np.logical_and(seg, gt_)
            az_merged = np.logical_or(az_merged, gt_thin)
            az_merged = np.logical_or(binary_closing(az_merged, iterations=2), az_merged)

        if check:
            v = napari.Viewer()
            v.add_image(raw)
            v.add_image(pred, blending="additive", visible=False)
            v.add_labels(seg, colormap={1: "blue"})
            v.add_labels(gt, colormap={1: "yellow"})
            v.add_labels(az_merged)
            v.title = f"{name}/{fname}"
            napari.run()

            print(f"gt_path {gt_path}")
            with h5py.File(gt_path, "a") as f:
                f.create_dataset(f"labels/az_merged_v{version}", data=az_merged, compression="lzf")

        else:
            print(f"gt_path {gt_path}")
            with h5py.File(gt_path, "a") as f:
                f.create_dataset(f"labels/az_merged_v{version}", data=az_merged, compression="lzf")
            '''with h5py.File(seg_path, "a") as f:
                f.create_dataset(f"labels/az_merged_v{version}", data=az_merged, compression="lzf")'''


def visualize_merge(args):
    for name in args.names:
        if "endbulb" in name:
            continue
        merge_az(name, args.version, check=True)


def copy_az(name, version):
    split_folder = get_split_folder(version)
    file_names = get_file_names(name, split_folder, split_names=["train", "val", "test"])
    _, gt_paths = get_paths(name, file_names, skip_seg=True)

    for gt_path in gt_paths:
        with h5py.File(gt_path, "a") as f:
            if "labels/az_merged" in f:
                continue
            az = f["labels/az"][:]
            f.create_dataset("labels/az_merged", data=az, compression="lzf")


def run_merge(args):
    for name in args.names:
        print("Merging", name)
        if "endbulb" in name:
            copy_az(name, args.version)
        else:
            merge_az(name, args.version, check=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--names", nargs="+", default=ALL_NAMES + ["endbulb_of_held_cropped"])
    parser.add_argument("--version", "-v", type=int, default=4)

    args = parser.parse_args()
    if args.visualize:
        visualize_merge(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
