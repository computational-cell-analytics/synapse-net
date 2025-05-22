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

def subtract_SVseg_from_AZseg(input_path, SV_path, output_path):
    """
    Modifies AZ segmentation by setting regions where SV segmentation is not zero to zero.

    Parameters:
    input_path (str): Path to the folder containing AZ segmentation H5 files.
    SV_path (str): Path to the folder containing corresponding SV segmentation H5 files.
    output_path (str): Path to save modified AZ segmentation H5 files.
    """

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all AZ segmentation files
    for file_name in os.listdir(input_path):
        if file_name.endswith(".h5"):
            az_file = os.path.join(input_path, file_name)
            sv_file = os.path.join(SV_path, file_name)

            # Check if corresponding SV segmentation file exists
            if os.path.exists(sv_file):
                with h5py.File(sv_file, "r") as f_sv, h5py.File(az_file, "r+") as f_az:
                    SV = f_sv["/vesicles/segment_from_combined_vesicles"][:]
                    AZ = f_az["/labels/az"][:]
                    raw = f_az["raw"][:]

                    # Modify AZ segmentation where SV segmentation is not zero
                    AZ[SV != 0] = 0

                # Save modified AZ segmentation
                output_file = os.path.join(output_path, file_name)
                with h5py.File(output_file, "a") as f:
                    f.create_dataset("raw", data=raw, compression="gzip")
                    f.create_dataset("labels/az", data=AZ, compression="gzip")

                print(f"Processed: {file_name}")

            else:
                print(f"Skipping {file_name}, corresponding SV file not found.")


def main():
    parser = argparse.ArgumentParser(description="Filter and process AZ data.")
    parser.add_argument("-i", "--input_path", required=True, type=str, help="Path to the root directory containing datasets.")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Path to the root directory for saving processed data.")
    parser.add_argument("-sv", "--SV_path", type=str, help="Path to the root directory that contains the SV segmentations.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    subtract_SVseg_from_AZseg(input_path, args.SV_path, output_path)

    #filter_all_azs(input_path, output_path)
    #process_all_azs(output_path)


if __name__ == "__main__":
    main()
