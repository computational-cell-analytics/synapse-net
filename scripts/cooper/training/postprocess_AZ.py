import os
from glob import glob
import argparse

import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_closing
from skimage.measure import label
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter
from synaptic_reconstruction.morphology import skeletonize_object



def filter_az(path, output_path):
    """Filter the active zone (AZ) data from the HDF5 file."""
    ds, fname = os.path.split(path)
    dataset_name = os.path.basename(ds)
    out_file_path = os.path.join(output_path, "postprocessed_AZ", dataset_name, fname)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    if os.path.exists(out_file_path):
        return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        az = f["AZ/segment_from_AZmodel_v3"][:]

    hmap = edge_filter(raw, sigma=1.0, method="sato", per_slice=True, n_threads=8)

    # Filter the active zone by combining a bunch of things:
    # 1. Find a mask with high values in the ridge filter.
    threshold_hmap = 0.5
    az_filtered = hmap > threshold_hmap
    # 2. Intersect it with the active zone predictions.
    az_filtered = np.logical_and(az_filtered, az)

    # Postprocessing of the filtered active zone:
    # 1. Apply connected components and only keep the largest component.
    az_filtered = label(az_filtered)
    ids, sizes = np.unique(az_filtered, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    az_filtered = (az_filtered == ids[np.argmax(sizes)]).astype("uint8")
    # 2. Apply binary closing.
    az_filtered = np.logical_or(az_filtered, binary_closing(az_filtered, iterations=4)).astype("uint8")

    # Save the result.
    with h5py.File(out_file_path, "a") as f:
        f.create_dataset("AZ/filtered_az", data=az_filtered, compression="gzip")


def process_az(path, view=False):
    """Skeletonize the filtered AZ data to obtain a 1D representation."""
    key = "AZ/thin_az"
    with h5py.File(path, "r") as f:
        if key in f and not view:
            return
        az_seg = f["AZ/filtered_az"][:]

    az_thin = skeletonize_object(az_seg)

    if view:
        import napari
        ds, fname = os.path.split(path)
        raw_path = os.path.join(ROOT, ds, fname)
        with h5py.File(raw_path, "r") as f:
            raw = f["raw"][:]
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(az_seg)
        v.add_labels(az_thin)
        napari.run()
    else:
        with h5py.File(path, "a") as f:
            f.create_dataset(key, data=az_thin, compression="gzip")


def filter_all_azs(input_path, output_path):
    """Apply filtering to all AZ data in the specified directory."""
    files = sorted(glob(os.path.join(input_path, "**/*.h5"), recursive=True))
    for ff in tqdm(files, desc="Filtering AZ segmentations"):
        filter_az(ff, output_path)


def process_all_azs(output_path):
    """Apply skeletonization to all filtered AZ data."""
    files = sorted(glob(os.path.join(output_path, "postprocessed_AZ", "**/*.h5"), recursive=True))
    for ff in tqdm(files, desc="Thinning AZ segmentations"):
        process_az(ff, view=False)


def main():
    parser = argparse.ArgumentParser(description="Filter and process AZ data.")
    parser.add_argument("input_path", type=str, help="Path to the root directory containing datasets.")
    parser.add_argument("output_path", type=str, help="Path to the root directory for saving processed data.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    filter_all_azs(input_path, output_path)
    process_all_azs(output_path)


if __name__ == "__main__":
    main()
