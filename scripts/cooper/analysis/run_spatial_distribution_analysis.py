import os
from glob import glob
import pandas as pd
import h5py
from tqdm import tqdm
from synaptic_reconstruction.distance_measurements import measure_segmentation_to_object_distances
import numpy as np

DATA_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/segmentation/for_spatial_distribution_analysis/final_Imig2014_seg/"  # noqa
PREDICTION_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/segmentation/for_spatial_distribution_analysis/final_Imig2014_seg/"  # noqa
RESULT_FOLDER = "./analysis_results/AZ_filtered_autoComp"
AZ_PATH = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/az_seg_filtered"
                

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
        compartment_mask = compartments == compartment_label
        vesicle_mask = vesicles > 0 

        intersection = np.logical_and(compartment_mask, vesicle_mask)
        overlap_count = np.sum(intersection)
        
        # Track the compartment with the most overlap in terms of voxel count
        if overlap_count > max_overlap_count:
            max_overlap_count = overlap_count
            best_compartment = compartment_label

    final_mask = compartments == best_compartment

    return final_mask

# We compute the distances for all vesicles in the AUTOMATIC SEGMENTED compartment masks to the AZ.
# We use different resolution, depending on dataset.
# The closest distance is calculated, i.e., the closest point on the outer membrane of the vesicle to the AZ.
def compute_per_vesicle_distance_to_AZ_autoComp(separate_AZseg=False):

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    resolution = (1.554,) * 3  # Change for each dataset #1.554 for Munc and snap #0.8681 for 04 dataset
    dataset_results = {}
    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))

    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        
        # Determine if the tomogram is 'CTRL' or 'DKO'
        category = "CTRL" if "CTRL" in fname else "DKO"
        
        if ds_name not in dataset_results:
            dataset_results[ds_name] = {'CTRL': {}, 'DKO': {}}

        if fname in dataset_results[ds_name][category]:
            continue


        # Load the vesicle segmentation from the predictions
        with h5py.File(tomo, "r") as f:
            segmentation = f["/vesicles/segment_from_combined_vesicles"][:]

            #Check if AZ seg is stored in a different tomo or same
            if separate_AZseg:
                print(f"using AZ segmentation from {AZ_PATH}")
                #Load the AZ segmentations
                AZ_path = os.path.join(AZ_PATH, ds_name, f"{fname}.h5")
                with h5py.File(AZ_path, "r") as f_AZ:
                    segmented_object = f_AZ["/thin_az"][:]
            else:
                segmented_object = f["/AZ/compartment_AZ_intersection"][:]

            #if AZ intersect is small, compartment seg didn't align with AZ so we use the normal AZ and not intersect
            if (segmented_object == 0).all() or np.sum(segmented_object == 1) < 2000:
                segmented_object = f["/AZ/segment_from_AZmodel_v3"][:]

        input_path = os.path.join(DATA_ROOT, ds_name, f"{fname}.h5")
        assert os.path.exists(input_path), input_path

        # Load the compartment mask from the tomogram
        with h5py.File(input_path, "r") as f:
            compartments  = f["/compartments/segment_from_3Dmodel_v2"][:]
        mask = get_compartment_with_max_overlap(compartments, segmentation)
        
        #if more than half of the vesicles (approximation, its checking pixel and not label) would get filtered by mask it means the compartment seg didn't work and thus we won't use the mask
        if np.sum(segmentation[mask == 0] > 0) > (0.5 * np.sum(segmentation > 0)):
            print("using no mask")
        else:
            segmentation[mask == 0] = 0

        distances, _, _, _ = measure_segmentation_to_object_distances(
            segmentation, segmented_object=segmented_object, resolution=resolution
        )

        # Add distances to the dataset dictionary under the appropriate category
        dataset_results[ds_name][category][fname] = distances

    # Save each dataset's results into separate CSV files for CTRL and DKO tomograms
    for ds_name, categories in dataset_results.items():
        for category, tomogram_data in categories.items():
            sorted_data = dict(sorted(tomogram_data.items()))  # Sort by tomogram names
            result_df = pd.DataFrame.from_dict(sorted_data, orient='index').transpose()
            output_path = os.path.join(RESULT_FOLDER, f"spatial_distribution_analysis_for_{ds_name}_{category}.csv")
            
            # Save the DataFrame to CSV
            result_df.to_csv(output_path, index=False)

# We compute the distances for all vesicles in the MANUALLY ANNOTATED compartment masks to the AZ.
# We use different resolution, depending on dataset.
# The closest distance is calculated, i.e., the closest point on the outer membrane of the vesicle to the AZ.
def compute_per_vesicle_distance_to_AZ_manComp():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    resolution = (1.554,) * 3  # Change for each dataset #1.554 for Munc and snap #0.8681 for 04 dataset
    dataset_results = {}
    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))
    
    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        
        # Determine if the tomogram is 'CTRL' or 'DKO'
        category = "CTRL" if "CTRL" in fname else "DKO"
        
        if ds_name not in dataset_results:
            dataset_results[ds_name] = {'CTRL': {}, 'DKO': {}}

        if fname in dataset_results[ds_name][category]:
            continue

        # Load the vesicle segmentation from the predictions
        with h5py.File(tomo, "r") as f:
            segmentation = f["/vesicles/segment_from_combined_vesicles"][:]
            segmented_object = f["/AZ/compartment_AZ_intersection_manComp"][:]

        input_path = os.path.join(DATA_ROOT, ds_name, f"{fname}.h5")
        assert os.path.exists(input_path), input_path

        # Load the compartment mask from the tomogram
        with h5py.File(input_path, "r") as f:
            mask  = f["/labels/compartment"][:]

        segmentation[mask == 0] = 0

        distances, _, _, _ = measure_segmentation_to_object_distances(
            segmentation, segmented_object=segmented_object, resolution=resolution
        )

        # Add distances to the dataset dictionary under the appropriate category
        dataset_results[ds_name][category][fname] = distances

    # Save each dataset's results into separate CSV files for CTRL and DKO tomograms
    for ds_name, categories in dataset_results.items():
        for category, tomogram_data in categories.items():
            sorted_data = dict(sorted(tomogram_data.items()))  # Sort by tomogram names
            result_df = pd.DataFrame.from_dict(sorted_data, orient='index').transpose()
            output_path = os.path.join(RESULT_FOLDER, f"spatial_distribution_analysis_for_{ds_name}_{category}.csv")
            
            # Save the DataFrame to CSV
            result_df.to_csv(output_path, index=False)
def main():
    compute_per_vesicle_distance_to_AZ_autoComp(separate_AZseg=False)
    #compute_per_vesicle_distance_to_AZ_manComp()



if __name__ == "__main__":
    main()
