import os
import argparse
from glob import glob
import json
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/user/muth9/u12095/synapse-net')
from synapse_net.training.domain_adaptation import mean_teacher_adaptation

OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/training/semisupervisedDA_cryo"

def _require_train_val_test_split(datasets, train_root, extension = "mrc"):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    def _train_val_test_split(names):
        train, test = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        _ratio = test_ratio / (test_ratio + val_ratio)
        val, test = train_test_split(test, test_size=_ratio)
        return train, val, test

    for ds in datasets:
        print(f"Processing dataset: {ds}")
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            print(f"Split file already exists: {split_path}")
            continue

        file_paths = sorted(glob(os.path.join(train_root, ds, f"*.{extension}")))
        file_names = [os.path.basename(path) for path in file_paths]

        train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)

def _require_train_val_split(datasets, train_root, extension = "mrc"):
    train_ratio, val_ratio = 0.8, 0.2

    def _train_val_split(names):
        train, val = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        return train, val

    for ds in datasets:
        print(f"Processing dataset: {ds}")
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            print(f"Split file already exists: {split_path}")
            continue

        file_paths = sorted(glob(os.path.join(train_root, ds, f"*.{extension}")))
        file_names = [os.path.basename(path) for path in file_paths]

        train, val = _train_val_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val}, f)

def get_paths(split, datasets, train_root, testset=True, extension = "mrc"):
    if testset:
        _require_train_val_test_split(datasets, train_root, extension)
    else:
        _require_train_val_split(datasets, train_root, extension)

    paths = []
    for ds in datasets:
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        with open(split_path) as f:
            names = json.load(f)[split]
        ds_paths = [os.path.join(train_root, ds, name) for name in names]
        assert all(os.path.exists(path) for path in ds_paths), f"Some paths do not exist in {ds_paths}"
        paths.extend(ds_paths)

    return paths

def vesicle_domain_adaptation(teacher_model, testset=True):
    # Adjustable parameters
    patch_shape = [48, 256, 256]
    model_name = "vesicle-semisupervisedDA-cryo-v1"
    model_root = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/models_v2/checkpoints/"
    checkpoint_path = os.path.join(model_root, teacher_model)

    unsupervised_train_root = "/mnt/lustre-emmy-hdd/usr/u12095/cryo-et"
    supervised_train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/cryoVesNet"

    unsupervised_datasets = ["from_portal"]
    unsupervised_train_paths = get_paths("train", datasets=unsupervised_datasets, train_root=unsupervised_train_root, testset=testset)
    unsupervised_val_paths = get_paths("val", datasets=unsupervised_datasets, train_root=unsupervised_train_root, testset=testset)

    supervised_datasets = ["exported"]
    supervised_train_paths = get_paths("train", datasets=supervised_datasets, train_root=supervised_train_root, testset=testset, extension = "h5")
    supervised_val_paths = get_paths("val", datasets=supervised_datasets, train_root=supervised_train_root, testset=testset, extension = "h5")

    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=unsupervised_train_paths,
        unsupervised_val_paths=unsupervised_val_paths,
        raw_key="data",
        supervised_train_paths=supervised_train_paths,
        supervised_val_paths=supervised_val_paths,
        raw_key_supervised = "raw",
        label_key="/labels/vesicles",
        patch_shape=patch_shape,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models/DA",
        source_checkpoint=checkpoint_path,
        confidence_threshold=0.75,
        n_iterations=int(1e3),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--teacher_model", required=True, help="Name of teacher model")
    parser.add_argument("-t", "--testset", action="store_false", help="Set to False if no testset should be created")
    args = parser.parse_args()

    vesicle_domain_adaptation(args.teacher_model, args.testset)

if __name__ == "__main__":
    main()
