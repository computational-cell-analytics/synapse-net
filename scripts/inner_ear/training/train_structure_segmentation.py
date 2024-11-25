import os
from glob import glob

import h5py
import numpy as np

from synaptic_reconstruction.training.supervised_training import supervised_training
from sklearn.model_selection import train_test_split
from torch_em.data.sampler import MinForegroundSampler
from tqdm import tqdm

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/inner_ear_data"
ROOT_OTHER_TOMOS = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/other_tomograms/"

LABEL_KEY = "labels/inner_ear_structures"
OTHER_NAMES = ["vesicle_pools", "tether", "rat"]


def get_other_paths(name):
    assert name in OTHER_NAMES, f"Invalid name {name}"
    if name == "vesicle_pools":
        folder = "01_vesicle_pools"
    elif name == "tether":
        folder = "02_tether"
    else:
        folder = "03_ratten_tomos"
    paths = sorted(glob(os.path.join(ROOT, folder, "*.h5")))
    return paths


def get_train_val_test_split(root):
    assert os.path.exists(root), root
    all_tomos = sorted(glob(os.path.join(root, "**/*.h5"), recursive=True))

    # Sort into train/test tomograms. We use the tomograms that have IMOD annotations for test.
    train_tomos, test_tomos = [], []
    for tomo in all_tomos:
        with h5py.File(tomo, "r") as f:
            if "labels/imod" in f:
                test_tomos.append(tomo)
            else:
                train_tomos.append(tomo)

    train_tomos, val_tomos = train_test_split(train_tomos, random_state=42, train_size=0.9)

    print("Number of train tomograms:", len(train_tomos))
    print("Number of val tomograms:", len(val_tomos))
    print("Number of test tomograms:", len(test_tomos))
    return train_tomos, val_tomos, test_tomos


def preprocess_labels(tomograms, structure_keys=("ribbon", "PD", "membrane")):
    nc = len(structure_keys)

    for tomo in tqdm(tomograms, desc="Preprocess labels"):
        with h5py.File(tomo, "a") as f:
            if LABEL_KEY in f:
                continue

            shape = f["raw"].shape
            labels = np.zeros((nc,) + shape, dtype="uint8")

            for channel, key in enumerate(structure_keys):
                assert f"labels/{key}" in f, f"{tomo} : {key}"
                this_labels = f[f"labels/{key}"][:]
                assert this_labels.shape == shape
                this_labels = (this_labels > 0).astype("uint8")
                labels[channel] = this_labels

            f.create_dataset(LABEL_KEY, data=labels, compression="gzip")


def noop(x):
    return x


def train_inner_ear_structures(train_tomograms, val_tomograms, name):
    patch_shape = (64, 512, 512)
    sampler = MinForegroundSampler(min_fraction=0.05, p_reject=0.95)
    supervised_training(
        name=name,
        train_paths=train_tomograms, val_paths=val_tomograms,
        label_key=LABEL_KEY, patch_shape=patch_shape, save_root=".",
        sampler=sampler, label_transform=noop, out_channels=3,
        with_label_channels=True, num_workers=12,
        n_samples_train=500, n_samples_val=25,
    )


def training_v1():
    train_tomograms, val_tomograms, _ = get_train_val_test_split(ROOT)
    preprocess_labels(train_tomograms)
    preprocess_labels(val_tomograms)
    train_inner_ear_structures(train_tomograms, val_tomograms, name="inner_ear_structure_model")


def training_v2():
    train_tomograms, val_tomograms, _ = get_train_val_test_split(ROOT)
    preprocess_labels(train_tomograms)
    for name in OTHER_NAMES:
        other_tomograms = get_other_paths(name)
        preprocess_labels(other_tomograms, structure_keys=("ribbons", "presynapse", "membrane"))
        train_tomograms.extend(other_tomograms)
    preprocess_labels(val_tomograms)
    train_inner_ear_structures(train_tomograms, val_tomograms, name="inner_ear_structure_model_v2")


def main():
    # training_v1()
    training_v2()


if __name__ == "__main__":
    main()
