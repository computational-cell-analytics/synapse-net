import json
import os


# The root folder which contains the new AZ training data.
INPUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data"
# The output folder for AZ predictions.
OUTPUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/AZ_prediction_new"

# The names of all datasets for which to run prediction / evaluation.
# This excludes 'endbulb_of_held_cropped', which is a duplicate of 'endbulb_of_held',
# which we don't evaluate on because of this.
ALL_NAMES = [
    "chemical_fixation", "endbulb_of_held", "stem", "tem"
]

# The translation of new dataset names to old dataset names.
NAME_TRANSLATION = {
    "chemical_fixation": ["12_chemical_fix_cryopreparation_minusSVseg_corrected"],
    "endbulb_of_held": ["wichmann_withAZ_rescaled_tomograms"],
    "stem": ["04_hoi_stem_examples_fidi_and_sarah_corrected_rescaled_tomograms"],
    "stem_cropped": ["04_hoi_stem_examples_minusSVseg_cropped_corrected_rescaled_tomograms",
                     "06_hoi_wt_stem750_fm_minusSVseg_cropped_corrected_rescaled_tomograms"],
    "tem": ["01data_withoutInvertedFiles_minusSVseg_corrected"],
}


# Get the paths to the files with raw data / ground-truth and the segmentation.
def get_paths(name, file_names, skip_seg=False):
    seg_paths, gt_paths = [], []
    for fname in file_names:
        if not skip_seg:
            seg_path = os.path.join(OUTPUT_ROOT, name, fname)
            assert os.path.exists(seg_path), seg_path
            seg_paths.append(seg_path)

        gt_path = os.path.join(INPUT_ROOT, name, fname)
        assert os.path.exists(gt_path), gt_path
        gt_paths.append(gt_path)

    return seg_paths, gt_paths


def get_file_names(name, split_folder, split_names):
    split_path = os.path.join(split_folder, f"split-{name}.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            splits = json.load(f)
        file_names = [fname for split in split_names for fname in splits[split]]

    else:
        old_names = NAME_TRANSLATION[name]
        file_names = []
        for old_name in old_names:
            split_path = os.path.join(split_folder, f"split-{old_name}.json")
            with open(split_path) as f:
                splits = json.load(f)
            this_file_names = [fname for split in split_names for fname in splits[split]]
            file_names.extend(this_file_names)
    return file_names


def get_split_folder(version):
    assert version in (3, 4, 5, 6, 7)
    if version == 3:
        split_folder = "splits"
    elif version == 6:
        split_folder= "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/splits"
    else:
        split_folder = "models_az_thin"
    return split_folder
