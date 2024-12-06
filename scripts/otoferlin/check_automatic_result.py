import os

import h5py
import napari
import numpy as np
import pandas as pd

from synapse_net.file_utils import read_mrc
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from common import get_all_tomograms, get_seg_path, get_colormaps


def _get_vesicle_pools(seg, assignment_path):
    assignments = pd.read_csv(assignment_path)
    pool_names = pd.unique(assignments.pool).tolist()
    pools = np.zeros_like(seg)

    pool_colors = get_colormaps()["pools"]
    colormap = {}
    for pool_id, pool_name in enumerate(pool_names, 1):
        pool_vesicle_ids = assignments[assignments.pool == pool_name].vesicle_id
        pool_mask = np.isin(seg, pool_vesicle_ids)
        pools[pool_mask] = pool_id
        colormap[pool_id] = pool_colors[pool_name]

    return pools, colormap


def check_automatic_result(mrc_path, version, use_clahe=False):

    seg_path = get_seg_path(mrc_path, version)

    segmentations, colormaps = {}, {}
    if os.path.exists(seg_path):
        with h5py.File(seg_path, "r") as f:
            g = f["segmentation"]
            for name, ds in g.items():
                segmentations[name] = ds[:]

    output_folder = os.path.split(seg_path)[0]
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    if os.path.exists(assignment_path):
        segmentations["pools"], colormaps["pools"] = _get_vesicle_pools(segmentations["vesicles"], assignment_path)

    tomogram, _ = read_mrc(mrc_path)
    if use_clahe:
        print("Run CLAHE ...")
        tomogram = equalize_adapthist(tomogram, clip_limit=0.03)
        print("... done")

    v = napari.Viewer()
    v.add_image(tomogram)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name, colormap=colormaps.get(name))
    v.title = os.path.basename(mrc_path)
    napari.run()


def main():
    # The version of automatic processing. Current versions:
    # 1: process everything with the synapse net default models
    version = 1
    tomograms = get_all_tomograms()
    for tomogram in tqdm(tomograms, desc="Visualize automatic segmentation results"):
        check_automatic_result(tomogram, version)


if __name__:
    main()
