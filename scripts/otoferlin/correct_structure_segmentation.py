import os
from glob import glob

import imageio.v3 as imageio
import h5py
import mrcfile
import napari
import numpy as np

# TODO refactor everything once things are merged
ROOT = "/home/ag-wichmann/data/otoferlin/tomograms"
if not os.path.exists(ROOT):
    ROOT = "./data/tomograms"

SEG_ROOT = "./segmentation/v2"


def correct_structure_segmentation(mrc_path):
    rel_path = os.path.relpath(mrc_path, ROOT)
    rel_folder, fname = os.path.split(rel_path)
    fname = os.path.splitext(fname)[0]
    seg_path = os.path.join(SEG_ROOT, rel_folder, f"{fname}.h5")

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = np.asarray(mrc.data[:])
    data = np.flip(data, axis=1)

    correction_folder = os.path.join(SEG_ROOT, rel_folder, "correction")

    names = ("ribbon", "PD", "membrane", "veiscles_postprocessed")
    segmentations = {}
    with h5py.File(seg_path, "r") as f:
        for name in names:
            correction_path = os.path.join(correction_folder, f"{name}.tif")
            if os.path.exists(correction_path):
                print("Loading segmentation for", name, "from", correction_path)
                segmentations[name] = imageio.imread(correction_path)
            else:
                segmentations[name] = f[f"segmentation/{name}"][:]
    color_maps = {
        "ribbon": {1: "red", None: "gray"},
        "PD": {1: "purple", None: "gray"},
        "membrane": {1: "magenta", None: "gray"},
    }

    v = napari.Viewer()
    v.add_image(data)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name, colormap=color_maps.get(name, None))
    v.title = fname
    napari.run()


def main():
    tomograms = glob(os.path.join(ROOT, "**", "*.mrc"), recursive=True)
    tomograms += glob(os.path.join(ROOT, "**", "*.rec"), recursive=True)
    tomograms = sorted(tomograms)

    for tomo in tomograms:
        correct_structure_segmentation(tomo)


if __name__ == "__main__":
    main()
