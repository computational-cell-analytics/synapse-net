import os

import numpy as np
import pandas as pd

from synapse_net.imod.to_imod import write_segmentation_to_imod,  write_segmentation_to_imod_as_points
from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, load_segmentations
from tqdm import tqdm


# TODO check if we need to remove offset from mrc
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

    # Load the pool assignments and export the pools to IMOD.
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    assignments = pd.read_csv(assignment_path)

    pools = pd.unique(assignments.pool)
    radius_factor = 1.0  # TODO!
    for pool in pools:
        export_path = os.path.join(export_folder, f"{pool}.mod")
        pool_ids = assignments[assignments.pool == pool].vesicle_ids
        pool_seg = vesicles.copy()
        pool_seg[~np.isin(pool_seg, pool_ids)] = 0
        write_segmentation_to_imod_as_points(
            mrc_path, pool_seg, export_path, min_radius=5, radius_factor=radius_factor
        )

    # TODO: read measurements for ribbon and PD volume / surface from IMOD.
    # - convert to meshes
    # - smooth the meshes
    # - run imodinfo to get the measurements
    measures = pd.DataFrame({
    })
    return measures


def main():
    force = False
    tomograms = get_all_tomograms()

    measurements = []
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        measures = export_tomogram(tomogram, force)
        measurements.append(measures)

    save_path = "./data/structure_measurements.xlsx"
    measurements = pd.concat(measurements)
    measurements.to_excel(save_path, index=False)


if __name__ == "__main__":
    main()
