# This is the code for the first analysis for the cooper data.
# Here, we only compute the vesicle numbers and size distributions for the STEM tomograms
# in the 04 dataset.

import os
from glob import glob

import pandas as pd
import h5py
from tqdm import tqdm
from synaptic_reconstruction.imod.to_imod import convert_segmentation_to_spheres

DATA_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014"  # noqa
PREDICTION_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/segmentation/for_spatial_distribution_analysis/20241102_TOMO_DATA_Imig2014"  # noqa
RESULT_FOLDER = "./analysis_results/analysis_1"


# We compute the sizes for all vesicles in the compartment masks.
# We use the same logic in the size computation as for the vesicle extraction to IMOD,
# including the radius correction factor.
# The number of vesicles is automatically computed as the length of the size list.
def compute_sizes_for_all_tomorams():
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    resolution = (0.8681,) * 3 #change for each dataset
    radius_factor = 1.3
    estimate_radius_2d = True

    tomograms = sorted(glob(os.path.join(PREDICTION_ROOT, "**/*.h5"), recursive=True))
    for tomo in tqdm(tomograms):
        ds_name, fname = os.path.split(tomo)
        ds_name = os.path.split(ds_name)[1]
        fname = os.path.splitext(fname)[0]
        output_path = os.path.join(RESULT_FOLDER, f"{ds_name}_{fname}.csv")
        if os.path.exists(output_path):
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

        result = pd.DataFrame({
            "dataset": [ds_name] * len(sizes),
            "tomogram": [fname] * len(sizes),
            "sizes": sizes
        })
        result.to_csv(output_path, index=False)


def main():
    compute_sizes_for_all_tomorams()


if __name__ == "__main__":
    main()
