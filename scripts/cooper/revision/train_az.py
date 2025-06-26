import argparse
import os
import json
from glob import glob

import torch_em

from sklearn.model_selection import train_test_split

from synapse_net.training import supervised_training, AZDistanceLabelTransform

TRAIN_ROOT = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data"
OUTPUT_ROOT = "./models_az_thin"


def _require_train_val_test_split(datasets):
    train_ratio, val_ratio, test_ratio = 0.60, 0.2, 0.2

    def _train_val_test_split(names):
        train, test = train_test_split(names, test_size=1 - train_ratio, shuffle=True)
        _ratio = test_ratio / (test_ratio + val_ratio)
        if len(test) == 2:
            val, test = test[:1], test[1:]
        else:
            val, test = train_test_split(test, test_size=_ratio)
        return train, val, test

    for ds in datasets:
        print(ds)
        split_path = os.path.join(OUTPUT_ROOT, f"split-{ds}.json")
        if os.path.exists(split_path):
            continue

        ds_root = os.path.join(TRAIN_ROOT, ds)
        assert os.path.exists(ds_root), ds_root
        file_paths = sorted(glob(os.path.join(ds_root, "*.h5")))
        file_names = [os.path.basename(path) for path in file_paths]

        train, val, test = _train_val_test_split(file_names)

        with open(split_path, "w") as f:
            json.dump({"train": train, "val": val, "test": test}, f)


def _require_train_val_split(datasets):
    train_ratio = 0.8

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
        assert len(ds_paths) > 0
        assert all(os.path.exists(path) for path in ds_paths)
        paths.extend(ds_paths)

    return paths


def train(key, ignore_label=None, use_distances=False, training_2D=False, testset=True, check=False):

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    datasets_with_testset_true = ["tem", "chemical_fixation", "stem", "endbulb_of_held"]
    datasets_with_testset_false = ["stem_cropped", "endbulb_of_held_cropped"]

    train_paths = get_paths("train", datasets=datasets_with_testset_true, testset=True)
    val_paths = get_paths("val", datasets=datasets_with_testset_true, testset=True)

    train_paths += get_paths("train", datasets=datasets_with_testset_false, testset=False)
    val_paths += get_paths("val", datasets=datasets_with_testset_false, testset=False)

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    # patch_shape = [48, 256, 256]
    patch_shape = [48, 384, 384]
    model_name = "v7"

    # checking for 2D training
    if training_2D:
        patch_shape = [1, 256, 256]
        model_name = "2D-AZ-model-v1"

    if use_distances:
        out_channels = 2
        label_transform = AZDistanceLabelTransform()
    else:
        out_channels = 1
        label_transform = torch_em.transform.label.labels_to_binary

    batch_size = 2
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        label_key=f"/labels/{key}",
        patch_shape=patch_shape, batch_size=batch_size,
        sampler=torch_em.data.sampler.MinInstanceSampler(min_num_instances=1, p_reject=0.85),
        n_samples_train=None, n_samples_val=100,
        check=check,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models/ConstantinAZ",
        n_iterations=int(2e5),
        ignore_label=ignore_label,
        label_transform=label_transform,
        out_channels=out_channels,
        # BCE_loss=False,
        # sigmoid_layer=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", help="Key ID that will be used by model in training", default="az_merged")
    parser.add_argument("-m", "--mask", type=int, default=None,
                        help="Mask ID that will be ignored by model in training")
    parser.add_argument("-2D", "--training_2D", action='store_true', help="Set to True for 2D training")
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()
    train(args.key, ignore_label=args.mask, training_2D=args.training_2D, testset=args.testset, check=args.check)


if __name__ == "__main__":
    main()
