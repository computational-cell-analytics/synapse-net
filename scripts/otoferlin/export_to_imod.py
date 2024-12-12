import os
from glob import glob

from pathlib import Path
from subprocess import run

import numpy as np
import pandas as pd

from tqdm import tqdm
from synapse_net.imod.to_imod import write_segmentation_to_imod, write_segmentation_to_imod_as_points
from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, load_segmentations


def check_imod(mrc_path, mod_path):
    run(["imod", mrc_path, mod_path])


def export_tomogram(mrc_path, force):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)

    # export_folder = os.path.join(output_folder, "imod")
    tomo_name = Path(mrc_path).stem
    export_folder = os.path.join(f"./imod/{tomo_name}")
    if os.path.exists(export_folder) and not force:
        return

    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]

    os.makedirs(export_folder, exist_ok=True)

    # Load the pool assignments and export the pools to IMOD.
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    assignments = pd.read_csv(assignment_path)

    colors = {
        "Docked-V": (255, 170, 127),  # (1, 0.666667, 0.498039)
        "RA-V": (0, 85, 0),  # (0, 0.333333, 0)
        "MP-V": (255, 170, 0),  # (1, 0.666667, 0)
        "ribbon": (255, 0, 0),
        "PD": (255, 0, 255),  # (1, 0, 1)
        "membrane": (255, 170, 255),  # 1, 0.666667, 1
    }

    pools = ['Docked-V', 'RA-V', 'MP-V']
    radius_factor = 0.85
    for pool in pools:
        export_path = os.path.join(export_folder, f"{pool}.mod")
        pool_ids = assignments[assignments.pool == pool].vesicle_id
        pool_seg = vesicles.copy()
        pool_seg[~np.isin(pool_seg, pool_ids)] = 0
        write_segmentation_to_imod_as_points(
            mrc_path, pool_seg, export_path, min_radius=5, radius_factor=radius_factor,
            color=colors.get(pool), name=pool,
        )
        # check_imod(mrc_path, export_path)

    # Export the structures to IMOD.
    for name in STRUCTURE_NAMES:
        export_path = os.path.join(export_folder, f"{name}.mod")
        color = colors.get(name)
        write_segmentation_to_imod(mrc_path, segmentations[name], export_path, color=color)
        # check_imod(mrc_path, export_path)

    # Join the model
    all_mod_files = sorted(glob(os.path.join(export_folder, "*.mod")))
    export_path = os.path.join(export_folder, f"{tomo_name}.mod")
    join_cmd = ["imodjoin"] + all_mod_files + [export_path]
    run(join_cmd)
    check_imod(mrc_path, export_path)


def main():
    force = True
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    tomograms_for_vis = [
        "Bl6_NtoTDAWT1_blockH_GridE4_1_rec.mrc",
        "Otof_TDAKO1blockA_GridN5_6_rec.mrc",
    ]
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        fname = os.path.basename(tomogram)
        if fname not in tomograms_for_vis:
            continue
        print("Exporting:", fname)
        export_tomogram(tomogram, force)


if __name__ == "__main__":
    main()
