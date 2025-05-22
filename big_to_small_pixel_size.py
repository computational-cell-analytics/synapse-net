import os
import numpy as np
import h5py
from glob import glob
from scipy.ndimage import zoom
from scipy.ndimage import label
from skimage.morphology import closing, ball

# Input and output folders
input_folder = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/AZ_data_after1stRevision/recorrected_length_of_AZ/wichmann_withAZ"
output_folder = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/AZ_data_after1stRevision/recorrected_length_of_AZ/wichmann_withAZ_right_rescaled"
os.makedirs(output_folder, exist_ok=True)

# Define scaling factors
old_pixel_size = np.array([1.75, 1.75, 1.75])
new_pixel_size = np.array([1.55, 1.55, 1.55])
scaling_factors = old_pixel_size / new_pixel_size

# Utility function to process segmentation
def rescale_and_fix_segmentation(segmentation, scaling_factors):
    """
    Rescale the segmentation and ensure labels are preserved.
    Args:
        segmentation (numpy.ndarray): The input segmentation array with integer labels.
        scaling_factors (list or array): Scaling factors for each axis.
    Returns:
        numpy.ndarray: Rescaled and hole-free segmentation with preserved labels.
    """
    # Rescale segmentation using nearest-neighbor interpolation
    rescaled_segmentation = zoom(segmentation, scaling_factors, order=0)

    # Initialize an array to hold the processed segmentation
    processed_segmentation = np.zeros_like(rescaled_segmentation)

    # Ensure no holes for each label
    unique_labels = np.unique(rescaled_segmentation)
    for label_id in unique_labels:
        if label_id == 0:  # Skip the background
            continue
        
        # Extract binary mask for the current label
        label_mask = rescaled_segmentation == label_id
        
        # Apply morphological closing to fill holes
        closed_mask = closing(label_mask, ball(1))
        
        # Add the processed label back to the output segmentation
        processed_segmentation[closed_mask] = label_id

    return processed_segmentation.astype(segmentation.dtype)


# Get all .h5 files in the specified input folder
h5_files = glob(os.path.join(input_folder, "*.h5"))
existing_files = {os.path.basename(f) for f in glob(os.path.join(output_folder, "*.h5"))}

for h5_file in h5_files:
    print(f"Processing {h5_file}...")

    if os.path.basename(h5_file) in existing_files:
        print(f"Skipping {h5_file} as it already exists in the output folder.")
        continue

    # Read data from the .h5 file
    with h5py.File(h5_file, "r") as f:
        raw = f["raw"][:]  # Assuming the dataset is named "raw"
        az = f["labels/az"][:]

    print(f"Original shape - raw: {raw.shape}; az: {az.shape}")

    # Process raw data (tomogram) with linear interpolation
    print("Rescaling raw data...")
    rescaled_raw = zoom(raw, scaling_factors, order=1)

    # Process az segmentation
    print("Rescaling and fixing az segmentation...")
    rescaled_az = rescale_and_fix_segmentation(az, scaling_factors)

    # Save the processed data to a new .h5 file
    output_path = os.path.join(output_folder, os.path.basename(h5_file))
    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=rescaled_raw, compression="gzip")
        f.create_dataset("labels/az", data=rescaled_az, compression="gzip")

    print(f"Saved rescaled data to {output_path}")

print("Processing complete. Rescaled files are saved in:", output_folder)
