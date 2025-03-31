import os
from glob import glob
import argparse
import json

from sklearn.model_selection import train_test_split

from synaptic_reconstruction.training import supervised_training
from synaptic_reconstruction.training import semisupervised_training

TRAIN_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/cryo-et/cryoVesNet/"
OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/for_revison/training_CryoVesNet"


def _require_train_val_test_split(datasets):
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

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

def train(key, ignore_label = None, training_2D = False, testset = True):

    datasets = [
    "exported"
]
    train_paths = get_paths("train", datasets=datasets, testset=testset)
    val_paths = get_paths("val", datasets=datasets, testset=testset)

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [48, 256, 256]
    model_name=f"3D-vesicles-model-for_revison_{key}"

    #checking for 2D training
    if training_2D:
        patch_shape = [1, 256, 256]
        model_name=f"2D-vesicles-model-new_postprocessing_{key}"
    
    batch_size = 4
    check = False

    supervised_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        label_key=f"/labels/vesicles",#/{key}",
        raw_key="raw",
        patch_shape=patch_shape, batch_size=batch_size,
        n_iterations=int(5e4),
        n_samples_train=None, n_samples_val=25,
        check=check,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/for_revison/models",
        ignore_label= ignore_label,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", required=True, help="Key ID that will be used by model in training")
    parser.add_argument("-m", "--mask", type=int, default=None, help="Mask ID that will be ignored by model in training")
    parser.add_argument("-2D", "--training_2D", action='store_true', help="Set to True for 2D training")
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.key, args.mask, args.training_2D, args.testset)


if __name__ == "__main__":
    main()
