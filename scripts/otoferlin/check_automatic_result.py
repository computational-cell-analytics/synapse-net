import os

import h5py
import napari
import numpy as np

from synapse_net.file_utils import read_mrc
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from common import get_all_tomograms, get_seg_path, get_colormaps


def check_automatic_result(mrc_path, version, use_clahe=False, center_crop=True, segmentation_group="segmentation"):
    tomogram, _ = read_mrc(mrc_path)
    if center_crop:
        halo = (50, 512, 512)
        bb = tuple(
            slice(max(sh // 2 - ha, 0), min(sh // 2 + ha, sh)) for sh, ha in zip(tomogram.shape, halo)
        )
        tomogram = tomogram[bb]
    else:
        bb = np.s_[:]

    if use_clahe:
        print("Run CLAHE ...")
        tomogram = equalize_adapthist(tomogram, clip_limit=0.03)
        print("... done")

    seg_path = get_seg_path(mrc_path, version)
    segmentations, colormaps = {}, {}
    if os.path.exists(seg_path):
        with h5py.File(seg_path, "r") as f:
            g = f[segmentation_group]
            for name, ds in g.items():
                segmentations[name] = ds[bb]
                colormaps[name] = get_colormaps().get(name, None)

    v = napari.Viewer()
    v.add_image(tomogram)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name, colormap=colormaps.get(name))
    v.title = os.path.basename(mrc_path)
    napari.run()


def main():
    version = 2
    tomograms = get_all_tomograms()
    for i, tomogram in tqdm(
        enumerate(tomograms), total=len(tomograms), desc="Visualize automatic segmentation results"
    ):
        print("Checking tomogram", tomogram)
        check_automatic_result(tomogram, version)
        # check_automatic_result(tomogram, version, segmentation_group="vesicles")
        # check_automatic_result(tomogram, version, segmentation_group="prediction")


if __name__:
    main()
