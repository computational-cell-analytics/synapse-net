import os
import argparse
from glob import glob
import json

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

TRAIN_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/2D_data"
OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/2D_DA_training_2Dcooper_v1"

def _require_train_val_test_split(datasets):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    if len(datasets) < 10:
        train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25

    def _train_val_test_split(names):
        train, test = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        _ratio = test_ratio / (test_ratio + val_ratio)
        val, test = train_test_split(test, test_size=_ratio)
        return train, val, test

    for ds in datasets:
        print(ds)
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        file_paths = sorted(glob(os.path.join(TRAIN_ROOT, ds, "*.h5")))
        file_names = [os.path.basename(path) for path in file_paths]

        train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)

def _require_train_val_split(datasets):
    train_ratio, val_ratio= 0.8, 0.2

    def _train_val_split(names):
        train, val = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        return train, val

    for ds in datasets:
        print(ds)
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        file_paths = sorted(glob(os.path.join(TRAIN_ROOT, ds, "*.h5")))
        file_names = [os.path.basename(path) for path in file_paths]

        train, val = _train_val_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val}, f)

def get_paths(split, datasets, testset=True):
    if testset:
        _require_train_val_test_split(datasets)
    else:
        _require_train_val_split(datasets)

    paths = []
    for ds in datasets:
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        with open(split_path) as f:
            names = json.load(f)[split]
        ds_paths = [os.path.join(TRAIN_ROOT, ds, name) for name in names]
        assert all(os.path.exists(path) for path in ds_paths)
        paths.extend(ds_paths)

    return paths

def vesicle_domain_adaptation(teacher_model, testset = True):

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    datasets = [
    "20241021_imig_2014_data_transfer_exported_grouped"
]
    train_paths = get_paths("train", datasets=datasets, testset=testset)
    val_paths = get_paths("val", datasets=datasets, testset=testset)
    
    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    #adjustable parameters
    patch_shape = [1, 256, 256] #2D
    model_name = "2D-vesicle-DA-2Dcooper-imig-v2"
    
    model_root = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/models_v2/checkpoints/"
    checkpoint_path = os.path.join(model_root, teacher_model)

    patch_shape = [256, 256] if any("maus" in dataset for dataset in datasets) else [1, 256, 256]

    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/DA_models",
        source_checkpoint=checkpoint_path,
        confidence_threshold=0.75,
        batch_size=8,
        n_iterations=int(1.5e4),
        n_samples_train=8000,
        n_samples_val=50,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--teacher_model", required=True, help="Name of teacher model")
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    
    vesicle_domain_adaptation(args.teacher_model, args.testset)


if __name__ == "__main__":
    main()
