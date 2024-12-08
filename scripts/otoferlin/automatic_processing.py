import os

import h5py
import numpy as np
import pandas as pd

from synapse_net.distance_measurements import measure_segmentation_to_object_distances, load_distances
from synapse_net.file_utils import read_mrc
from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.tools.util import get_model, compute_scale_from_voxel_size, _segment_ribbon_AZ
from tqdm import tqdm

from common import STRUCTURE_NAMES, get_all_tomograms, get_seg_path, get_adapted_model


def _get_center_crop(input_):
    halo_xy = (600, 600)
    bb_xy = tuple(
        slice(max(sh // 2 - ha, 0), min(sh // 2 + ha, sh)) for sh, ha in zip(input_.shape[1:], halo_xy)
    )
    bb = (np.s_[:],) + bb_xy
    return bb, input_.shape


def _get_tiling():
    tile = {"x": 768, "y": 768, "z": 64}
    halo = {"x": 128, "y": 128, "z": 8}
    return {"tile": tile, "halo": halo}


def process_vesicles(mrc_path, output_path, process_center_crop):
    key = "segmentation/vesicles"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if key in f:
                return

    input_, voxel_size = read_mrc(mrc_path)
    if process_center_crop:
        bb, full_shape = _get_center_crop(input_)
        input_ = input_[bb]

    model = get_adapted_model()
    scale = compute_scale_from_voxel_size(voxel_size, "ribbon")
    print("Rescaling volume for vesicle segmentation with factor:", scale)
    tiling = _get_tiling()
    segmentation = segment_vesicles(input_, model=model, scale=scale, tiling=tiling)

    if process_center_crop:
        full_seg = np.zeros(full_shape, dtype=segmentation.dtype)
        full_seg[bb] = segmentation
        segmentation = full_seg

    with h5py.File(output_path, "a") as f:
        f.create_dataset(key, data=segmentation, compression="gzip")


def process_ribbon_structures(mrc_path, output_path, process_center_crop):
    key = "segmentation/ribbon"
    with h5py.File(output_path, "r") as f:
        if key in f:
            return
        vesicles = f["segmentation/vesicles"][:]

    input_, voxel_size = read_mrc(mrc_path)
    if process_center_crop:
        bb, full_shape = _get_center_crop(input_)
        input_ = input_[bb]

    model_name = "ribbon"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    tiling = _get_tiling()
    segmentations, predictions = _segment_ribbon_AZ(
        input_, model, tiling=tiling, scale=scale, verbose=True, extra_segmentation=vesicles,
        return_predictions=True, n_slices_exclude=5,
    )

    if process_center_crop:
        for name, seg in segmentations:
            full_seg = np.zeros(full_shape, dtype=seg.dtype)
            full_seg[bb] = seg
            segmentations[name] = full_seg
        for name, pred in predictions:
            full_pred = np.zeros(full_shape, dtype=seg.dtype)
            full_pred[bb] = pred
            predictions[name] = full_pred

    with h5py.File(output_path, "a") as f:
        for name, seg in segmentations.items():
            f.create_dataset(f"segmentation/{name}", data=seg, compression="gzip")
            f.create_dataset(f"prediction/{name}", data=predictions[name], compression="gzip")


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


def process_tomogram(mrc_path):
    output_path = get_seg_path(mrc_path)
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    process_center_crop = True

    process_vesicles(mrc_path, output_path, process_center_crop)
    process_ribbon_structures(mrc_path, output_path, process_center_crop)
    return
    # TODO vesicle post-processing:
    # snap to boundaries?
    # remove vesicles in ribbon

    measure_distances(mrc_path, output_path, output_folder)
    assign_vesicle_pools(output_folder)


def main():
    tomograms = get_all_tomograms()
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        process_tomogram(tomogram)


if __name__:
    main()
