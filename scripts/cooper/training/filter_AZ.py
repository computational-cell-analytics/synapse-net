import os
import h5py
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label

def process_labels(label_file_path, erosion_structure=None, dilation_structure=None):
    """
    Process the labels: perform erosion, find the largest connected component,
    and perform dilation on it.

    Args:
        label_file_path (str): Path to the HDF5 file containing the label data.
        erosion_structure (ndarray, optional): Structuring element for erosion.
        dilation_structure (ndarray, optional): Structuring element for dilation.

    Returns:
        None: The processed data is saved back into the HDF5 file under a new key.
    """
    with h5py.File(label_file_path, "r+") as label_file:
        # Read the ground truth data
        gt = label_file["/labels/filtered_az"][:]

        # Perform binary erosion
        eroded = binary_erosion(gt, structure=erosion_structure)

        # Label connected components
        labeled_array, num_features = label(eroded)
        
        # Identify the largest connected component
        if num_features > 0:
            largest_component_label = np.argmax(np.bincount(labeled_array.flat, weights=eroded.flat)[1:]) + 1
            largest_component = (labeled_array == largest_component_label)
        else:
            largest_component = np.zeros_like(gt, dtype=bool)

        # Perform binary dilation on the largest connected component
        dilated = binary_dilation(largest_component, structure=dilation_structure)

        # Save the result back into the HDF5 file
        if "labels/erosion_filtered_az" in label_file:
            del label_file["labels/erosion_filtered_az"]  # Remove if it already exists
        label_file.create_dataset("labels/erosion_filtered_az", data=dilated.astype(np.uint8), compression="gzip")

def process_folder(folder_path, erosion_structure=None, dilation_structure=None):
    """
    Process all HDF5 files in a folder.

    Args:
        folder_path (str): Path to the folder containing HDF5 files.
        erosion_structure (ndarray, optional): Structuring element for erosion.
        dilation_structure (ndarray, optional): Structuring element for dilation.

    Returns:
        None
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".h5") or file_name.endswith(".hdf5"):
            label_file_path = os.path.join(folder_path, file_name)
            print(f"Processing {label_file_path}...")
            process_labels(label_file_path, erosion_structure, dilation_structure)

# Example usage
if __name__ == "__main__":
    folder_path = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/training_AZ_v2/postprocessed_AZ/12_chemical_fix_cryopreparation"  # Replace with the path to your folder
    erosion_structure = np.ones((3, 3, 3))  # Example structuring element
    dilation_structure = np.ones((3, 3, 3))  # Example structuring element
    process_folder(folder_path, erosion_structure, dilation_structure)
