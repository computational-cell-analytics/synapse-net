# This is the code for the first analysis for the cooper data.
# Here, we only compute the vesicle numbers and size distributions for the STEM tomograms
# in the 04 dataset.

import os
from glob import glob

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from synaptic_reconstruction.imod.to_imod import convert_segmentation_to_spheres

DATA_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/exported/"  # noqa
PREDICTION_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/segmentation/for_spatial_distribution_analysis/final_Imig2014_seg_manComp"  # noqa
RESULT_FOLDER = "./analysis_results/AZ_intersect_manualCompartment"

def get_compartment_with_max_overlap(compartments, vesicles):
    """
    Given 3D numpy arrays of compartments and vesicles, this function returns a binary mask
    of the compartment with the most overlap with vesicles based on the number of overlapping voxels.
    
    Parameters:
    compartments (numpy.ndarray): 3D array of compartment labels.
    vesicles (numpy.ndarray): 3D array of vesicle labels or binary mask.

    Returns:
    numpy.ndarray: Binary mask of the compartment with the most overlap with vesicles.
    """
    
    unique_compartments = np.unique(compartments)
    if 0 in unique_compartments:
        unique_compartments = unique_compartments[unique_compartments != 0]

    max_overlap_count = 0
    best_compartment = None

    # Iterate over each compartment and calculate the overlap with vesicles
    for compartment_label in unique_compartments:
        # Create a binary mask for the current compartment
        compartment_mask = compartments == compartment_label
        vesicle_mask = vesicles > 0 

        intersection = np.logical_and(compartment_mask, vesicle_mask)
        
        # Calculate the number of overlapping voxels
        overlap_count = np.sum(intersection)
        
        # Track the compartment with the most overlap in terms of voxel count
        if overlap_count > max_overlap_count:
            max_overlap_count = overlap_count
            best_compartment = compartment_label

    # Create the final mask for the compartment with the most overlap
    final_mask = compartments == best_compartment

    return final_mask

# We compute the sizes for all vesicles in the compartment masks.
# We use the same logic in the size computation as for the vesicle extraction to IMOD,
# including the radius correction factor.
# The number of vesicles is automatically computed as the length of the size list.
def compute_sizes_for_all_tomorams_manComp():
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    resolution = (1.554,) * 3  # Change for each dataset #1.554 for Munc and snap #0.8681 for 04 dataset
    radius_factor = 1
    estimate_radius_2d = True

    # Dictionary to hold the results for each dataset and category (CTRL or DKO)
    dataset_results = {}

    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))
    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        
        # Determine if the tomogram is 'CTRL' or 'DKO'
        category = "CTRL" if "CTRL" in fname else "DKO"
        
        # Initialize a new dictionary entry for each dataset and category if not already present
        if ds_name not in dataset_results:
            dataset_results[ds_name] = {'CTRL': {}, 'DKO': {}}
        
        # Skip if this tomogram already exists in the dataset dictionary
        if fname in dataset_results[ds_name][category]:
            continue
            
        # Load the vesicle segmentation from the predictions.
        with h5py.File(tomo, "r") as f:
            segmentation = f["/vesicles/segment_from_combined_vesicles"][:]

        input_path = os.path.join(DATA_ROOT, ds_name, f"{fname}.h5")
        assert os.path.exists(input_path), input_path
        # Load the compartment mask from the tomogram
        with h5py.File(input_path, "r") as f:
            mask = f["labels/compartment"][:]

        segmentation[mask == 0] = 0
        _, sizes = convert_segmentation_to_spheres(
            segmentation, resolution=resolution, radius_factor=radius_factor, estimate_radius_2d=estimate_radius_2d
        )

        # Add sizes to the dataset dictionary under the appropriate category
        dataset_results[ds_name][category][fname] = sizes

    # Save each dataset's results into separate CSV files for CTRL and DKO tomograms
    for ds_name, categories in dataset_results.items():
        for category, tomogram_data in categories.items():
            # Sort tomograms by name within the category
            sorted_data = dict(sorted(tomogram_data.items()))  # Sort by tomogram names
            result_df = pd.DataFrame.from_dict(sorted_data, orient='index').transpose()
            
            # Define the output file path
            output_path = os.path.join(RESULT_FOLDER, f"size_analysis_for_{ds_name}_{category}_rf1.csv")
            
            # Save the DataFrame to CSV
            result_df.to_csv(output_path, index=False)

def compute_sizes_for_all_tomorams_autoComp():
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    resolution = (1.554,) * 3  # Change for each dataset #1.554 for Munc and snap #0.8681 for 04 dataset
    radius_factor = 1
    estimate_radius_2d = True

    # Dictionary to hold the results for each dataset and category (CTRL or DKO)
    dataset_results = {}

    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))
    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        
        # Determine if the tomogram is 'CTRL' or 'DKO'
        category = "CTRL" if "CTRL" in fname else "DKO"
        
        # Initialize a new dictionary entry for each dataset and category if not already present
        if ds_name not in dataset_results:
            dataset_results[ds_name] = {'CTRL': {}, 'DKO': {}}
        
        # Skip if this tomogram already exists in the dataset dictionary
        if fname in dataset_results[ds_name][category]:
            continue

        # Load the vesicle segmentation from the predictions.
        with h5py.File(tomo, "r") as f:
            segmentation = f["/vesicles/segment_from_combined_vesicles"][:]

        input_path = os.path.join(DATA_ROOT, ds_name, f"{fname}.h5")
        assert os.path.exists(input_path), input_path
        # Load the compartment mask from the tomogram
        with h5py.File(input_path, "r") as f:
            compartments  = f["/compartments/segment_from_3Dmodel_v2"][:]
        mask = get_compartment_with_max_overlap(compartments, segmentation)
        
        # if more than half of the vesicles (approximation, its checking pixel and not label) would get filtered by mask it means the compartment seg didn't work and thus we won't use the mask
        if np.sum(segmentation[mask == 0] > 0) > (0.5 * np.sum(segmentation > 0)):
            print(f"using no mask for {tomo}")
        else:
            segmentation[mask == 0] = 0
        _, sizes = convert_segmentation_to_spheres(
            segmentation, resolution=resolution, radius_factor=radius_factor, estimate_radius_2d=estimate_radius_2d
        )

        # Add sizes to the dataset dictionary under the appropriate category
        dataset_results[ds_name][category][fname] = sizes

    # Save each dataset's results into separate CSV files for CTRL and DKO tomograms
    for ds_name, categories in dataset_results.items():
        for category, tomogram_data in categories.items():
            # Sort tomograms by name within the category
            sorted_data = dict(sorted(tomogram_data.items()))  # Sort by tomogram names
            result_df = pd.DataFrame.from_dict(sorted_data, orient='index').transpose()
            
            # Define the output file path
            output_path = os.path.join(RESULT_FOLDER, f"size_analysis_for_{ds_name}_{category}_rf1.csv")
            
            # Save the DataFrame to CSV
            result_df.to_csv(output_path, index=False)

def main():
    compute_sizes_for_all_tomorams_manComp()
    #compute_sizes_for_all_tomorams_autoComp()


if __name__ == "__main__":
    main()
