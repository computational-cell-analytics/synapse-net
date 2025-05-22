import os
from glob import glob
import h5py
import numpy as np
from skimage.transform import rescale


def rescale_h5_files(input_folder, output_folder, current_resolution, target_resolution):
    # Compute the scale factor: target / current
    scale_factor = current_resolution / target_resolution

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all .h5 files in the input folder
    h5_files = glob(os.path.join(input_folder, "*.h5"))
    existing_files = {os.path.basename(f) for f in glob(os.path.join(output_folder, "*.h5"))}

    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        output_path = os.path.join(output_folder, filename)

        if filename in existing_files:
            print(f"Skipping {filename}, already exists in output.")
            continue

        print(f"Processing {filename}...")

        with h5py.File(h5_file, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        # Rescale in 3D
        raw_rescaled = rescale(
            raw,
            scale=(scale_factor, scale_factor, scale_factor),
            order=3,
            preserve_range=True,
            anti_aliasing=True
        ).astype(raw.dtype)

        az_rescaled = rescale(
            az,
            scale=(scale_factor, scale_factor, scale_factor),
            order=0,  # Nearest neighbor for label maps
            preserve_range=True,
            anti_aliasing=False
        ).astype(az.dtype)

        # Save to new .h5 file
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("raw", data=raw_rescaled, compression="gzip")
            f_out.create_dataset("labels/az", data=az_rescaled, compression="gzip")

        print(f"Saved rescaled data to {output_path}")


def main():
    input_folder = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/AZ_data_after1stRevision/recorrected_length_of_AZ/wichmann_withAZ"
    output_folder = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/AZ_data_after1stRevision/recorrected_length_of_AZ/wichmann_withAZ_rescaled_tomograms"

    current_resolution = 0.8681  # nm
    target_resolution = 1.55     # nm

    rescale_h5_files(input_folder, output_folder, current_resolution, target_resolution)


if __name__ == "__main__":
    main()
