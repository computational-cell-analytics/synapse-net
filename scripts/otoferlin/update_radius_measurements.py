import os
import pandas as pd
from pool_assignments_and_measurements import _measure_radii

from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, load_segmentations
from tqdm import tqdm


def update_radii(mrc_path):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    radii = _measure_radii(seg_path)
    
    pool_assignments = pd.read_csv(assignment_path)
    assert len(radii) == len(pool_assignments)
    pool_assignments["radius"] = radii
    pool_assignments["diameter"] = 2 * radii

    pool_assignments.to_csv(assignment_path, index=False)


def main():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        update_radii(tomogram)


if __name__:
    main()
