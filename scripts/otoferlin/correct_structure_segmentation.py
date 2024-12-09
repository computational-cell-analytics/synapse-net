import os
from pathlib import Path

import imageio.v3 as imageio
import h5py
import napari

from synapse_net.file_utils import read_mrc
from common import get_all_tomograms, get_seg_path


def correct_structure_segmentation(mrc_path):
    seg_path = get_seg_path(mrc_path)

    data, _ = read_mrc(mrc_path)
    correction_folder = os.path.join(os.path.split(seg_path)[0], "correction")
    fname = Path(mrc_path).stem

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
    tomograms = get_all_tomograms()
    for tomo in tomograms:
        correct_structure_segmentation(tomo)


if __name__ == "__main__":
    main()
