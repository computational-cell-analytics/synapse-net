import os

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd
from magicgui import magicgui

from synapse_net.file_utils import read_mrc
from synapse_net.distance_measurements import load_distances
from skimage.measure import regionprops
from common import load_segmentations, get_seg_path, get_all_tomograms, get_colormaps, STRUCTURE_NAMES
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# FIXME: adding vesicles to pool doesn't work / messes with color map
def _create_pool_layer(seg, assignment_path):
    assignments = pd.read_csv(assignment_path)
    pools = np.zeros_like(seg)

    pool_colors = get_colormaps()["pools"]
    colormap = {None: "gray", 0: (0, 0, 0, 0)}

    # Sorting of floats and ints by np.unique is weird. We better don't trust unique here
    # It should not matter if one of the pools is empty.
    pool_names = ["RA-V", "MP-V", "Docked-V"]

    for pool_id, pool_name in enumerate(pool_names, 1):
        if not isinstance(pool_name, str) and np.isnan(pool_name):
            continue
        pool_vesicle_ids = assignments[assignments.pool == pool_name].vesicle_id.values
        pool_mask = np.isin(seg, pool_vesicle_ids)
        pools[pool_mask] = pool_id
        colormap[pool_id] = pool_colors[pool_name]

    return pools, colormap, assignments


def _update_assignments(vesicles, pool_correction, assignment_path):
    old_assignments = pd.read_csv(assignment_path)
    props = regionprops(vesicles, pool_correction)

    val_to_pool = {0: 0, 1: "RA-V", 2: "MP-V", 3: "Docked-V", 4: None}
    corrected_pools = {prop.label: val_to_pool[int(prop.max_intensity)] for prop in props}

    new_assignments = []
    for _, row in old_assignments.iterrows():
        vesicle_id = row.vesicle_id
        corrected_pool = corrected_pools[vesicle_id]
        if corrected_pool != 0:
            row.pool = corrected_pool
        new_assignments.append(row)
    new_assignments = pd.DataFrame(new_assignments)
    new_assignments.to_csv(assignment_path, index=False)


def _create_outlier_mask(assignments, vesicles, output_folder):
    distances = {}
    for name in STRUCTURE_NAMES:
        dist, _, _, ids = load_distances(os.path.join(output_folder, "distances", f"{name}.npz"))
        distances[name] = {vid: dist for vid, dist in zip(ids, dist)}

    pool_criteria = {
        "RA-V": {"ribbon": 80},
        "MP-V": {"PD": 100, "membrane": 50},
        "Docked-V": {"PD": 100, "membrane": 2},
    }
    
    vesicle_ids = assignments.vesicle_id.values
    outlier_ids = []
    for pool in ("RA-V", "MP-V", "Docked-V"):
        pool_ids = assignments[assignments.pool == pool].vesicle_id.values
        for name in STRUCTURE_NAMES:
            min_dist = pool_criteria[pool].get(name)
            if min_dist is None:
                continue
            dist = distances[name]
            assert len(dist) == len(vesicle_ids)
            pool_outliers = [vid for vid in pool_ids if dist[vid] > min_dist]
            if pool_outliers:
                print("Pool:", pool, ":", name, ":", len(pool_outliers))
            outlier_ids.extend(pool_outliers)

    outlier_ids = np.unique(outlier_ids)
    outlier_mask = np.isin(vesicles, outlier_ids).astype("uint8")
    return outlier_mask


def correct_vesicle_pools(mrc_path, show_outliers, skip_if_no_outlier=False):
    seg_path = get_seg_path(mrc_path)

    output_folder = os.path.split(seg_path)[0]
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    if not os.path.exists(assignment_path):
        print("Skip", seg_path, "due to missing assignments")
        return

    data, _ = read_mrc(mrc_path)
    segmentations = load_segmentations(seg_path, verbose=False)
    vesicles = segmentations["vesicles"]

    colormaps = get_colormaps()
    pool_colors = colormaps["pools"]
    correction_colors = {
        1: pool_colors["RA-V"], 2: pool_colors["MP-V"], 3: pool_colors["Docked-V"], 4: "Gray", None: "Gray"
    }

    vesicle_pools, pool_colors, assignments = _create_pool_layer(vesicles, assignment_path)
    if show_outliers:
        outlier_mask = _create_outlier_mask(assignments, vesicles, output_folder)
    else:
        outlier_mask = None

    if skip_if_no_outlier and outlier_mask.sum() == 0:
        return

    pool_correction_path = os.path.join(output_folder, "correction", "pool_correction.tif")
    os.makedirs(os.path.join(output_folder, "correction"), exist_ok=True)
    if os.path.exists(pool_correction_path):
        pool_correction = imageio.imread(pool_correction_path)
    else:
        pool_correction = np.zeros_like(vesicles)

    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(vesicle_pools, colormap=pool_colors)
    v.add_labels(pool_correction, colormap=correction_colors)
    v.add_labels(vesicles, visible=False)
    for name in STRUCTURE_NAMES:
        # v.add_labels(segmentations[name], name=name, visible=False, colormap=colormaps[name])
        v.add_labels(segmentations[name], name=name, visible=False)

    if outlier_mask is not None:
        v.add_labels(outlier_mask)

    @magicgui(call_button="Update Pools")
    def update_pools(viewer: napari.Viewer):
        pool_data = viewer.layers["vesicle_pools"].data
        vesicles = viewer.layers["vesicles"].data
        pool_correction = viewer.layers["pool_correction"].data
        _update_assignments(vesicles, pool_correction, assignment_path)
        pool_data, pool_colors, _ = _create_pool_layer(vesicles, assignment_path)
        viewer.layers["vesicle_pools"].data = pool_data
        viewer.layers["vesicle_pools"].colormap = pool_colors

    v.window.add_dock_widget(update_pools)
    v.title = os.path.basename(mrc_path)

    napari.run()


def main():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for tomo in tqdm(tomograms):
        correct_vesicle_pools(tomo, show_outliers=True, skip_if_no_outlier=True)


if __name__ == "__main__":
    main()
