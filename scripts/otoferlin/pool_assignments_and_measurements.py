import os

import numpy as np
import pandas as pd

from synapse_net.distance_measurements import measure_segmentation_to_object_distances, load_distances
from synapse_net.file_utils import read_mrc
from synapse_net.imod.to_imod import convert_segmentation_to_spheres
from skimage.measure import label
from tqdm import tqdm

from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, load_segmentations


def ensure_labeled(vesicles):
    n_ids = len(np.unique(vesicles))
    n_ids_labeled = len(np.unique(label(vesicles)))
    assert n_ids == n_ids_labeled, f"{n_ids}, {n_ids_labeled}"


def measure_distances(mrc_path, seg_path, output_folder, force):
    result_folder = os.path.join(output_folder, "distances")
    if os.path.exists(result_folder) and not force:
        return

    # Get the voxel size.
    _, voxel_size = read_mrc(mrc_path)
    resolution = tuple(voxel_size[ax] for ax in "zyx")

    # Load the segmentations.
    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]
    ensure_labeled(vesicles)
    structures = {name: segmentations[name] for name in STRUCTURE_NAMES}

    # Measure all the object distances.
    os.makedirs(result_folder, exist_ok=True)
    for name, seg in structures.items():
        if seg.sum() == 0:
            print(name, "was not found, skipping the distance computation.")
            continue
        print("Compute vesicle distances to", name)
        save_path = os.path.join(result_folder, f"{name}.npz")
        measure_segmentation_to_object_distances(vesicles, seg, save_path=save_path, resolution=resolution)


def _measure_radii(seg_path):
    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]
    # The radius factor of 0.85 yields the best fit to vesicles in IMOD.
    _, radii = convert_segmentation_to_spheres(vesicles, radius_factor=0.85)
    return np.array(radii)


def assign_vesicle_pools_and_measure_radii(seg_path, output_folder, force):
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    if os.path.exists(assignment_path) and not force:
        return

    distance_folder = os.path.join(output_folder, "distances")
    distance_paths = {name: os.path.join(distance_folder, f"{name}.npz") for name in STRUCTURE_NAMES}
    if not all(os.path.exists(path) for path in distance_paths.values()):
        print("Skip vesicle pool assignment, because some distances are missing.")
        print("This is probably due to the fact that the corresponding structures were not found.")
        return
    distances = {name: load_distances(path) for name, path in distance_paths.items()}

    # The distance criteria.
    rav_ribbon_distance = 80  # nm
    mpv_pd_distance = 100  # nm
    mpv_mem_distance = 50  # nm
    docked_pd_distance = 100  # nm
    docked_mem_distance = 2  # nm

    rav_distances, seg_ids = distances["ribbon"][0], np.array(distances["ribbon"][-1])
    rav_ids = seg_ids[rav_distances < rav_ribbon_distance]

    pd_distances, mem_distances = distances["PD"][0], distances["membrane"][0]
    assert len(pd_distances) == len(mem_distances) == len(rav_distances)

    mpv_ids = seg_ids[np.logical_and(pd_distances < mpv_pd_distance, mem_distances < mpv_mem_distance)]
    docked_ids = seg_ids[np.logical_and(pd_distances < docked_pd_distance, mem_distances < docked_mem_distance)]

    # Create a dictionary to map vesicle ids to their corresponding pool.
    # (RA-V get's over-written by MP-V, which is correct).
    pool_assignments = {vid: "RA-V" for vid in rav_ids}
    pool_assignments.update({vid: "MP-V" for vid in mpv_ids})
    pool_assignments.update({vid: "Docked-V" for vid in docked_ids})

    pool_values = [pool_assignments.get(vid, None) for vid in seg_ids]
    radii = _measure_radii(seg_path)
    assert len(radii) == len(pool_values)

    pool_assignments = pd.DataFrame({
        "vesicle_id": seg_ids,
        "pool": pool_values,
        "radius": radii,
        "diameter": 2 * radii,
    })
    pool_assignments.to_csv(assignment_path, index=False)


def process_tomogram(mrc_path, force):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)

    # Measure the distances.
    measure_distances(mrc_path, seg_path, output_folder, force)

    # Assign the vesicle pools.
    assign_vesicle_pools_and_measure_radii(seg_path, output_folder, force)

    # The surface area / volume for ribbon and PD will be done in a separate script.


def main():
    force = True
    tomograms = get_all_tomograms(restrict_to_good_tomos=True, restrict_to_nachgeb=True)
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        process_tomogram(tomogram, force)


if __name__ == "__main__":
    main()
