import os

import h5py
import numpy as np
import pandas as pd

from synapse_net.distance_measurements import measure_segmentation_to_object_distances, load_distances
from synapse_net.file_utils import read_mrc
from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.tools.util import get_model, compute_scale_from_voxel_size, _segment_ribbon_AZ
from tqdm import tqdm

from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path


def process_vesicles(mrc_path, output_path, version):
    key = "segmentation/vesicles"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if key in f:
                return

    input_, voxel_size = read_mrc(mrc_path)

    model_name = "vesicles_3d"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    print("Rescaling volume for vesicle segmentation with factor:", scale)
    segmentation = segment_vesicles(input_, model=model, scale=scale)

    with h5py.File(output_path, "a") as f:
        f.create_dataset(key, data=segmentation, compression="gzip")


def process_ribbon_structures(mrc_path, output_path, version):
    key = "segmentation/ribbon"
    with h5py.File(output_path, "r") as f:
        if key in f:
            return
        vesicles = f["segmentation/vesicles"][:]

    input_, voxel_size = read_mrc(mrc_path)
    model_name = "ribbon"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    segmentations = _segment_ribbon_AZ(
        input_, model, tiling=None, scale=scale, verbose=True, extra_segmentation=vesicles
    )

    with h5py.File(output_path, "a") as f:
        for name, seg in segmentations.items():
            f.create_dataset(f"segmentation/{name}", data=seg, compression="gzip")


def measure_distances(mrc_path, seg_path, output_folder):
    result_folder = os.path.join(output_folder, "distances")
    if os.path.exists(result_folder):
        return

    # Get the voxel size.
    _, voxel_size = read_mrc(mrc_path)
    resolution = tuple(voxel_size[ax] for ax in "zyx")

    # Load the segmentations.
    with h5py.File(seg_path, "r") as f:
        g = f["segmentation"]
        vesicles = g["vesicles"][:]
        structures = {name: g[name][:] for name in STRUCTURE_NAMES}

    # Measure all the object distances.
    os.makedirs(result_folder, exist_ok=True)
    for name, seg in structures.items():
        if seg.sum() == 0:
            print(name, "was not found, skipping the distance computation.")
            continue
        print("Compute vesicle distances to", name)
        save_path = os.path.join(result_folder, f"{name}.npz")
        measure_segmentation_to_object_distances(vesicles, seg, save_path=save_path, resolution=resolution)


def assign_vesicle_pools(output_folder):
    assignment_path = os.path.join(output_folder, "vesicle_pools.csv")
    if os.path.exists(assignment_path):
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

    pool_assignments = pd.DataFrame({
        "vesicle_id": list(pool_assignments.keys()),
        "pool": list(pool_assignments.values()),
    })
    pool_assignments.to_csv(assignment_path, index=False)


def process_tomogram(mrc_path, version):
    output_path = get_seg_path(mrc_path, version)
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    process_vesicles(mrc_path, output_path, version)
    process_ribbon_structures(mrc_path, output_path, version)

    measure_distances(mrc_path, output_path, output_folder)
    assign_vesicle_pools(output_folder)


def main():
    # The version of automatic processing. Current versions:
    # 1: process everything with the synapse net default models
    version = 1
    tomograms = get_all_tomograms()
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        process_tomogram(tomogram, version)


if __name__:
    main()
