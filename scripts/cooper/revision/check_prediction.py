import argparse
import os

import h5py
import napari
from common import ALL_NAMES, get_file_names, get_split_folder, get_paths


def check_predictions(name, split, version):
    split_folder = get_split_folder(version)
    file_names = get_file_names(name, split_folder, split_names=[split])
    seg_paths, gt_paths = get_paths(name, file_names)

    for seg_path, gt_path in zip(seg_paths, gt_paths):

        with h5py.File(gt_path, "r") as f:
            raw = f["raw"][:]
            gt = f["labels/az"][:] if version == 3 else f["labels/az_thin"][:]

        with h5py.File(seg_path) as f:
            pred_key = f"predictions/az/v{version}"
            seg_key = f"predictions/az/seg_v{version}"
            pred = f[pred_key][:]
            seg = f[seg_key][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(pred, blending="additive")
        v.add_labels(gt)
        v.add_labels(seg)
        v.title = f"{name}/{os.path.basename(seg_path)}"
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--names", nargs="+", default=ALL_NAMES)
    args = parser.parse_args()

    for name in args.names:
        check_predictions(name, args.split, args.version)


if __name__ == "__main__":
    main()
