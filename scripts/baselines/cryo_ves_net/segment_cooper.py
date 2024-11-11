import os
from glob import glob

from common import apply_cryo_vesnet

INPUT_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets"  # noqa
OUTPUT_ROOT = "./predictions/cooper"  # noqa

RESOLUTIONS = {
    "01_hoi_maus_2020_incomplete": (1.554, 1.554, 1.554),
    "02_hcc_nanogold": (1.2, 1.2, 1.2),
    "03_hog_cs1sy7": (1.24, 1.24, 1.24),
    "05_stem750_sv_training": (0.868, 0.868, 0.868),
    "07_hoi_s1sy7_tem250_ihgp": (1.24, 1.24, 1.24),
    "10_tem_single_release": (1.24, 1.24, 1.24),
    "11_tem_multiple_release": (1.24, 1.24, 1.24),
    "12_chemical_fix_cryopreparation": (1.554, 1.554, 1.554)
}


def segment_dataset(ds_name, resolution):
    input_folder = os.path.join(INPUT_ROOT, ds_name)
    output_folder = os.path.join(OUTPUT_ROOT, ds_name)

    n_inputs = len(glob(os.path.join(input_folder, "*.h5")))
    n_outputs = len(glob(os.path.join(output_folder, "*.h5")))
    if n_inputs == n_outputs:
        print(ds_name, "is already processed")
        return

    apply_cryo_vesnet(
        input_folder, output_folder, pattern="*.h5", input_key="raw", resolution=resolution, nested=True
    )


def segment_04():
    ds_name = "04"
    input_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/04Dataset_for_vesicle_eval"  # noqa
    output_folder = os.path.join(OUTPUT_ROOT, ds_name)

    n_inputs = len(glob(os.path.join(input_folder, "**/*.h5"), recursive=True))
    n_outputs = len(glob(os.path.join(output_folder, "**/*.h5"), recursive=True))
    if n_inputs == n_outputs:
        print(ds_name, "is already processed")
        return

    resolution = (8.681, 8.681, 8.681)
    apply_cryo_vesnet(
        input_folder, output_folder, pattern="*.h5", input_key="raw", resolution=resolution, nested=True,
        mask_folder=input_folder, mask_key="labels/compartment"
    )


def main():
    segment_04()
    for ds_name, resolution in RESOLUTIONS.items():
        resolution = tuple(res * 10 for res in resolution)
        segment_dataset(ds_name, resolution)


if __name__ == "__main__":
    main()
