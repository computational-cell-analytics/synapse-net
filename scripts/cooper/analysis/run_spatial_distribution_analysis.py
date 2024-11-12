import os
from glob import glob
import pandas as pd
import h5py
from tqdm import tqdm
from synaptic_reconstruction.distance_measurements import measure_segmentation_to_object_distances

DATA_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014"  # noqa
PREDICTION_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/segmentation/for_spatial_distribution_analysis/20241102_TOMO_DATA_Imig2014"  # noqa
RESULT_FOLDER = "./analysis_results"


# We compute the distances for all vesicles in the compartment masks to the AZ.
# We use the same different resolution, depending on dataset.
# The closest distance is calculated, i.e., the closest point on the outer membrane of the vesicle to the AZ.
def compute_sizes_for_all_tomorams():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    
    resolution = (1.554,) * 3  # Change for each dataset #1.554 for Munc and snap #0.8681 for 04 dataset
    
    # Dictionary to hold the results for each dataset
    dataset_results = {}
    
    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))
    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        
        # Initialize a new dictionary entry for each dataset if not already present
        if ds_name not in dataset_results:
            dataset_results[ds_name] = {}
        
        # Skip if this tomogram already exists in the dataset dictionary
        if fname in dataset_results[ds_name]:
            continue

        # Load the vesicle segmentation from the predictions
        with h5py.File(tomo, "r") as f:
            segmentation = f["/vesicles/segment_from_combined_vesicles"][:]
            segmented_object = f["/AZ/segment_from_AZmodel_v3"][:]

        input_path = os.path.join(DATA_ROOT, ds_name, f"{fname}.h5")
        assert os.path.exists(input_path), input_path

        # Load the compartment mask from the tomogram
        with h5py.File(input_path, "r") as f:
            mask = f["labels/compartment"][:]

        segmentation[mask == 0] = 0
        distances, _, _, _ = measure_segmentation_to_object_distances(
            segmentation, segmented_object=segmented_object, resolution=resolution
        )

        # Add distances to the dataset dictionary under the tomogram name
        dataset_results[ds_name][fname] = distances

    # Save each dataset's results to a single CSV file
    for ds_name, tomogram_data in dataset_results.items():
        # Create a DataFrame where each column is a tomogram's distances
        result_df = pd.DataFrame.from_dict(tomogram_data, orient='index').transpose()
        
        # Define the output file path
        output_path = os.path.join(RESULT_FOLDER, f"spatial_distribution_analysis_for_{ds_name}.csv")
        
        # Save the DataFrame to CSV
        result_df.to_csv(output_path, index=False)

        
def main():
    compute_sizes_for_all_tomorams()


if __name__ == "__main__":
    main()
