import os
import sys
from glob import glob

import imageio.v3 as imageio
import h5py
import napari
import numpy as np
import pandas

from synaptic_reconstruction.distance_measurements import create_object_distance_lines
from synaptic_reconstruction.file_utils import get_data_path
from synaptic_reconstruction.tools.distance_measurement import _downsample

from elf.io import open_file
from tqdm import tqdm

sys.path.append("processing")


def get_distance_visualization(tomo, segmentations, distance_paths, vesicle_ids, scale):
    tomo = _downsample(tomo, scale=scale)
    segmentations = {
        k: _downsample(v, is_seg=True, scale=scale, target_shape=tomo.shape)
        for k, v in segmentations.items()
    }

    ribbon_lines, _ = create_object_distance_lines(distance_paths["ribbon"], seg_ids=vesicle_ids, scale=scale)
    pd_lines, _ = create_object_distance_lines(distance_paths["PD"], seg_ids=vesicle_ids, scale=scale)
    membrane_lines, _ = create_object_distance_lines(distance_paths["membrane"], seg_ids=vesicle_ids, scale=scale)

    distance_lines = {
        "ribbon_distances": ribbon_lines,
        "pd_distances": pd_lines,
        "membrane_distances": membrane_lines
    }
    return tomo, segmentations, distance_lines


def create_vesicle_pools(vesicles, result_path):
    vesicle_pools = np.zeros_like(vesicles)

    assignment_result = pandas.read_excel(result_path)
    vesicle_ids = assignment_result["id"].values
    pool_assignments = assignment_result["pool"].values
    pool_assignments = {vid: pool for vid, pool in zip(vesicle_ids, pool_assignments)}

    color_map = {
        "RA-V": (0, 0.33, 0),
        "MP-V": (1, 0.66, 0),
        "Docked-V": (1, 0.66, 0.5),
    }
    colors = {}
    for pool_id, pool_name in enumerate(("RA-V", "MP-V", "Docked-V"), 1):
        ves_ids_pool = [vid for vid, pname in pool_assignments.items() if pname == pool_name]
        pool_mask = np.isin(vesicles, ves_ids_pool)
        vesicle_pools[pool_mask] = vesicles[pool_mask]
        colors.update({vid: color_map[pool_name] for vid in ves_ids_pool})

    vesicle_ids = np.sort(vesicle_ids)
    return vesicle_pools, vesicle_ids, {"vesicle_pools": colors}


def _load_segmentation(correction_file, seg_file):
    if os.path.exists(correction_file):
        seg = imageio.imread(correction_file)
    else:
        with h5py.File(seg_file, "r") as f:
            seg = f["segmentation"][:] if "segmentation" in f else f["prediction"][:]
    if seg.dtype == np.dtype("uint64"):
        seg = seg.astype("uint32")
    return seg


def visualize_folder(folder, segmentation_version, visualize_distances, binning):
    from parse_table import _match_correction_folder, _match_correction_file

    raw_path = get_data_path(folder)
    with open_file(raw_path, "r") as f:
        tomo = f["data"][:]

    if segmentation_version is None:

        v = napari.Viewer()
        v.add_image(tomo)
        v.title = folder
        napari.run()

    else:
        correction_folder = _match_correction_folder(folder)
        seg_folder = os.path.join(folder, "automatisch", f"v{segmentation_version}")
        seg_files = glob(os.path.join(seg_folder, "*.h5"))
        if len(seg_files) == 0:
            print("No segmentations for", folder, "skipping!")

        if os.path.exists(os.path.join(correction_folder, "measurements.xlsx")):
            result_path = os.path.join(correction_folder, "measurements.xlsx")
            distance_folder = os.path.join(correction_folder, "distances")
            correction = "Displaying corrected vesicle pools for"
        else:
            result_path = os.path.join(seg_folder, "measurements.xlsx")
            distance_folder = os.path.join(seg_folder, "distances")
            correction = "Displaying UNCORRECTED vesicle pools"

        segmentations = {}
        for seg_file in seg_files:
            seg_name = seg_file.split("_")[-1].rstrip(".h5")
            correction_file = _match_correction_file(correction_folder, seg_name)
            seg = _load_segmentation(correction_file, seg_file)
            segmentations[seg_name] = seg

        if os.path.exists(result_path):
            segmentations["vesicle_pools"], vesicle_ids, colors = create_vesicle_pools(
                segmentations["vesicles"], result_path
            )
        else:
            colors = {}
            vesicle_ids = None

        distance_files = {
            name: os.path.join(distance_folder, f"{name}.npz") for name in ["ribbon", "PD", "membrane"]
        }
        have_distances = all(os.path.exists(path) for path in distance_files.values())

        if have_distances and visualize_distances:
            assert vesicle_ids is not None
            tomo, segmentations, distance_lines = get_distance_visualization(
                tomo, segmentations, distance_files, vesicle_ids, scale=binning
            )
        else:
            distance_lines = {}

        v = napari.Viewer()
        v.add_image(tomo)
        for name, seg in segmentations.items():
            v.add_labels(seg, name=name, color=colors.get(name, None))

        for name, lines in distance_lines.items():
            v.add_shapes(lines, shape_type="line", name=name, visible=False)

        v.title = f"{correction}: {folder}"
        napari.run()


def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def _get_bin_factor(is_old, binning):
    if isint(binning):
        return int(binning)
    elif binning == "none":
        return None
    else:
        assert binning == "auto"
        return 2 if is_old else 3


def visualize_all_data(
    data_root, table,
    segmentation_version=None, check_micro=None,
    visualize_distances=False, skip_iteration=None,
    binning="auto",
):
    assert check_micro in ["new", "old", "both", None]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        if skip_iteration is not None and i < skip_iteration:
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "alt" and check_micro in ("old", "both", None):
            binning_ = _get_bin_factor(True, binning)
            visualize_folder(folder, segmentation_version, visualize_distances, binning_)

        elif micro == "neu" and ("new", "both", None):
            binning_ = _get_bin_factor(False, binning)
            visualize_folder(folder, segmentation_version, visualize_distances, binning_)

        elif micro == "beides":
            if check_micro in ("old", "both", None):
                binning_ = _get_bin_factor(True, binning)
                visualize_folder(folder, segmentation_version, visualize_distances, binning_)
            if check_micro in ("new", "both", None):
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                binning_ = _get_bin_factor(False, binning)
                visualize_folder(
                    folder_new, segmentation_version, visualize_distances, binning_
                )


def main():
    import argparse
    from parse_table import parse_table, get_data_root

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iteration", default=None, type=int)
    parser.add_argument("-m", "--microscope", default=None)
    parser.add_argument("-d", "--visualize_distances", action="store_false")
    parser.add_argument("-b", "--binning", default="auto")
    args = parser.parse_args()
    assert args.microscope in (None, "both", "old", "new")

    binning = args.binning
    assert (binning in ("none", "auto") or isint(binning))

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")

    table = parse_table(table_path, data_root)

    segmentation_version = 2

    visualize_all_data(
        data_root, table,
        segmentation_version=segmentation_version, check_micro=args.microscope,
        visualize_distances=args.visualize_distances, skip_iteration=args.iteration,
        binning=binning,
    )


# Tomos With Artifacts:
# Analyse/WT strong stim/Mouse 1/modiolar/14/Emb71M1aGridA3sec3mod14.rec
# Analyse/WT strong stim/Mouse 1/modiolar/18/Emb71M1aGridA1sec1mod3.rec
# Analyse/WT strong stim/Mouse 1/modiolar/8/Emb71M1aGridA1sec1mod2.rec

# New Tomos for Annotation:
# WT Strong stim/Mouse 1/ modiolar/1
# -> ribbon segmentation + PD
# WT Contriol/Mouse 2/ modiolar/5
# -> membrane segmentation
if __name__ == "__main__":
    main()
