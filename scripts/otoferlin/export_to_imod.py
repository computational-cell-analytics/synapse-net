import os
from subprocess import run

import numpy as np
import pandas as pd

from tqdm import tqdm
from synapse_net.imod.to_imod import write_segmentation_to_imod,  write_segmentation_to_imod_as_points
from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, load_segmentations


def check_imod(mrc_path, mod_path):
    run(["imod", mrc_path, mod_path])


def export_tomogram(mrc_path, force):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)

    export_folder = os.path.join(output_folder, "imod")
    if os.path.exists(export_folder) and not force:
        return

    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]

    os.makedirs(export_folder, exist_ok=True)

    # Export the structures to IMOD.
    for name in STRUCTURE_NAMES:
        export_path = os.path.join(export_folder, f"{name}.mod")
        write_segmentation_to_imod(mrc_path, segmentations[name], export_path)
        # check_imod(mrc_path, export_path)

    # Load the pool assignments and export the pools to IMOD.
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    assignments = pd.read_csv(assignment_path)

    pools = pd.unique(assignments.pool)
    # TODO: discuss this with Clara, not sure how to handle this with the irregular vesicles.
    radius_factor = 1.0
    for pool in pools:
        export_path = os.path.join(export_folder, f"{pool}.mod")
        pool_ids = assignments[assignments.pool == pool].vesicle_id
        pool_seg = vesicles.copy()
        pool_seg[~np.isin(pool_seg, pool_ids)] = 0
        write_segmentation_to_imod_as_points(
            mrc_path, pool_seg, export_path, min_radius=5, radius_factor=radius_factor
        )
        # check_imod(mrc_path, export_path)


def main():
    force = True
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)

    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        export_tomogram(tomogram, force)


if __name__ == "__main__":
    main()
