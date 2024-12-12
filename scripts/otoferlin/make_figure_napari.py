import os

import napari
import numpy as np
import pandas as pd

from synapse_net.file_utils import read_mrc
from common import get_all_tomograms, get_seg_path, load_segmentations, STRUCTURE_NAMES


colors = {
    "Docked-V": (255, 170, 127),  # (1, 0.666667, 0.498039)
    "RA-V": (0, 85, 0),  # (0, 0.333333, 0)
    "MP-V": (255, 170, 0),  # (1, 0.666667, 0)
    "ribbon": (255, 0, 0),
    "PD": (255, 0, 255),  # (1, 0, 1)
    "membrane": (255, 170, 255),  # 1, 0.666667, 1
}


def plot_napari(mrc_path):
    data, voxel_size = read_mrc(mrc_path)
    voxel_size = tuple(voxel_size[ax] for ax in "zyx")

    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]

    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    assignments = pd.read_csv(assignment_path)

    pools = np.zeros_like(vesicles)
    pool_names = ["RA-V", "MP-V", "Docked-V"]

    pool_colors = {None: (0, 0, 0)}
    for pool_id, pool_name in enumerate(pool_names, 1):
        pool_vesicle_ids = assignments[assignments.pool == pool_name].vesicle_id.values
        pool_mask = np.isin(vesicles, pool_vesicle_ids)
        pools[pool_mask] = pool_id
        color = colors.get(pool_name)
        color = tuple(c / float(255) for c in color)
        pool_colors[pool_id] = color

    v = napari.Viewer()
    v.add_image(data, scale=voxel_size)
    v.add_labels(pools, colormap=pool_colors, scale=voxel_size)
    for name in STRUCTURE_NAMES:
        color = colors[name]
        color = tuple(c / float(255) for c in color)
        cmap = {1: color, None: (0, 0, 0)}
        v.add_labels(segmentations[name], colormap=cmap, scale=voxel_size, name=name)
    v.scale_bar.visible = True
    v.scale_bar.unit = "nm"
    v.scale_bar.font_size = 18
    v.title = os.path.basename(mrc_path)
    napari.run()


def main():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    tomograms_for_vis = [
        "Bl6_NtoTDAWT1_blockH_GridE4_1_rec.mrc",
        "Otof_TDAKO1blockA_GridN5_6_rec.mrc",
    ]
    for tomogram in tomograms:
        fname = os.path.basename(tomogram)
        if fname not in tomograms_for_vis:
            continue
        plot_napari(tomogram)


main()
