import os
import pandas as pd
from synapse_net.distance_measurements import load_distances

from common import get_all_tomograms, get_seg_path, load_table
        

def _add_one_to_assignment(mrc_path):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    
    assignments = pd.read_csv(assignment_path)
    assignments["ribbon_id"] = len(assignments) * [1]
    assignments.to_csv(assignment_path, index=False)


def _update_assignments(mrc_path, num_ribbon):
    print(mrc_path)
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    distance_path = os.path.join(output_folder, "distances", "ribbon.npz")

    _, _, _, seg_ids, object_ids = load_distances(distance_path, return_object_ids=True)
    assert all(obj in range(1, num_ribbon + 1) for obj in object_ids)
    
    assignments = pd.read_csv(assignment_path)
    assert len(assignments) == len(object_ids)
    assert (seg_ids == assignments.vesicle_id.values).all()
    assignments["ribbon_id"] = object_ids
    assignments.to_csv(assignment_path, index=False)


def process_tomogram(mrc_path):
    table = load_table()
    table = table[table["File name"] == os.path.basename(mrc_path)]
    assert len(table) == 1
    num_ribbon = int(table["#ribbons"].values[0])
    assert num_ribbon in (1, 2)

    if num_ribbon == 1:
        _add_one_to_assignment(mrc_path)
    else:
        _update_assignments(mrc_path, num_ribbon)


def main():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for tomogram in tomograms:
        process_tomogram(tomogram)


if __name__ == "__main__":
    main()
