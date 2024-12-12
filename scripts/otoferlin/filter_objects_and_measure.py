import os
from tqdm import tqdm

import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from common import get_all_tomograms, get_seg_path, load_table, load_segmentations, STRUCTURE_NAMES
from synapse_net.distance_measurements import measure_segmentation_to_object_distances, load_distances
from synapse_net.file_utils import read_mrc


def _filter_n_objects(segmentation, num_objects):
    # Create individual objects for all disconnected pieces.
    segmentation = label(segmentation)
    # Find object ids and sizes, excluding background.
    ids, sizes = np.unique(segmentation, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    # Only keep the biggest 'num_objects' objects.
    keep_ids = ids[np.argsort(sizes)[::-1]][:num_objects]
    segmentation[~np.isin(segmentation, keep_ids)] = 0
    # Relabel the segmentation sequentially.
    segmentation, _, _ = relabel_sequential(segmentation)
    # Ensure that we have the correct number of objects.
    n_ids = int(segmentation.max())
    assert n_ids == num_objects
    return segmentation


def filter_and_measure(mrc_path, seg_path, output_folder, force):
    result_folder = os.path.join(output_folder, "distances")
    if os.path.exists(result_folder) and not force:
        return

    # Load the table to find out how many ribbons / PDs we have here.
    table = load_table()
    table = table[table["File name"] == os.path.basename(mrc_path)]
    assert len(table) == 1

    num_ribbon = int(table["#ribbons"].values[0])
    num_pd = int(table["PD?"].values[0])

    segmentations = load_segmentations(seg_path)
    vesicles = segmentations["vesicles"]
    structures = {name: segmentations[name] for name in STRUCTURE_NAMES}

    # Filter the ribbon and the PD.
    print("Filtering number of ribbons:", num_ribbon)
    structures["ribbon"] = _filter_n_objects(structures["ribbon"], num_ribbon)
    print("Filtering number of PDs:", num_pd)
    structures["PD"] = _filter_n_objects(structures["PD"], num_pd)

    _, resolution = read_mrc(mrc_path)
    resolution = [resolution[ax] for ax in "zyx"]

    # Measure all the object distances.
    for name in ("ribbon", "PD"):
        seg = structures[name]
        assert seg.sum() != 0, name
        print("Compute vesicle distances to", name)
        save_path = os.path.join(result_folder, f"{name}.npz")
        measure_segmentation_to_object_distances(vesicles, seg, save_path=save_path, resolution=resolution)


def process_tomogram(mrc_path, force):
    seg_path = get_seg_path(mrc_path)
    output_folder = os.path.split(seg_path)[0]
    assert os.path.exists(output_folder)

    # Measure the distances.
    filter_and_measure(mrc_path, seg_path, output_folder, force)


def main():
    force = True
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        process_tomogram(tomogram, force)


if __name__:
    main()
