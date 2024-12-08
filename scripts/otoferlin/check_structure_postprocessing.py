import os

import h5py
import napari
import numpy as np

from synapse_net.file_utils import read_mrc
from tqdm import tqdm

from common import get_seg_path, get_all_tomograms, STRUCTURE_NAMES


def check_structure_postprocessing(mrc_path, center_crop=True):
    tomogram, _ = read_mrc(mrc_path)
    if center_crop:
        halo = (50, 512, 512)
        bb = tuple(
            slice(max(sh // 2 - ha, 0), min(sh // 2 + ha, sh)) for sh, ha in zip(tomogram.shape, halo)
        )
        tomogram = tomogram[bb]
    else:
        bb = np.s_[:]

    seg_path = get_seg_path(mrc_path)
    assert os.path.exists(seg_path)

    segmentations, predictions, colormaps = {}, {}, {}
    with h5py.File(seg_path, "r") as f:
        g = f["segmentation"]
        for name in STRUCTURE_NAMES:
            segmentations[name] = g[name][bb]
        g = f["prediction"]
        for name in STRUCTURE_NAMES:
            predictions[name] = g[name][bb]

    v = napari.Viewer()
    v.add_image(tomogram)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name, colormap=colormaps.get(name))
    for name, seg in predictions.items():
        v.add_labels(seg, name=name, colormap=colormaps.get(name))
    v.title = os.path.basename(mrc_path)
    napari.run()


def main():
    tomograms = get_all_tomograms()
    for i, tomogram in tqdm(enumerate(tomograms), total=len(tomograms), desc="Check structure postproc"):
        check_structure_postprocessing(tomogram)


if __name__:
    main()
