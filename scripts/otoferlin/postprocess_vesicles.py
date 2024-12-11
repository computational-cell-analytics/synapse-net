import os
from pathlib import Path
from shutil import copyfile

import imageio.v3 as imageio
import napari
import h5py

from skimage.measure import label
from tqdm import tqdm

from common import get_all_tomograms, get_seg_path
from automatic_processing import postprocess_vesicles

TOMOS = [
    "Otof_TDAKO2blockC_GridE2_1",
    "Otof_TDAKO1blockA_GridN5_3",
    "Otof_TDAKO1blockA_GridN5_5",
    "Bl6_NtoTDAWT1_blockH_GridG2_3",
]


def postprocess(mrc_path, process_center_crop):
    output_path = get_seg_path(mrc_path)
    copyfile(output_path, output_path + ".bkp")
    postprocess_vesicles(
        mrc_path, output_path, process_center_crop=process_center_crop, force=False
    )

    with h5py.File(output_path, "r") as f:
        ves = f["segmentation/veiscles_postprocessed"][:]

    v = napari.Viewer()
    v.add_labels(ves)
    napari.run()


# Postprocess vesicles in specific tomograms, where this initially
# failed due to wrong structure segmentations.
def redo_initial_postprocessing():
    tomograms = get_all_tomograms()
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        fname = Path(tomogram).stem
        if fname not in TOMOS:
            continue
        print("Postprocessing", fname)
        postprocess(tomogram, process_center_crop=True)


def label_all_vesicles():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for mrc_path in tqdm(tomograms, desc="Process tomograms"):
        output_path = get_seg_path(mrc_path)
        output_folder = os.path.split(output_path)[0]
        vesicle_path = os.path.join(output_folder, "correction", "veiscles_postprocessed.tif")
        assert os.path.exists(vesicle_path), vesicle_path
        copyfile(vesicle_path, vesicle_path + ".bkp")
        vesicles = imageio.imread(vesicle_path)
        vesicles = label(vesicles)
        imageio.imwrite(vesicle_path, vesicles, compression="zlib")


def main():
    # redo_initial_postprocessing()
    # Label all vesicle corrections to make sure everyone has its own id
    label_all_vesicles()


if __name__:
    main()
