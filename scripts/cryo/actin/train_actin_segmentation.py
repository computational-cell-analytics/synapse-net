import numpy as np

import torch_em
from synapse_net.training import supervised_training
from torch_em.data.sampler import MinForegroundSampler

import os
from pathlib import Path

def train_actin_deepict_v3():
    """Train a network for actin segmentation on the deepict dataset.
    Tomograms are split into subvolumes, which are assigned to train, val, or test sets.
    """
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/")
    TRAIN_DIR = PARENT_DIR / "train"
    VAL_DIR = PARENT_DIR / "val"

    train_paths = [str(p) for p in TRAIN_DIR.glob("*.h5")]
    val_paths = [str(p) for p in VAL_DIR.glob("*.h5")]

    patch_shape = (64, 384, 384)
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-deepict-v3",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        n_iterations=int(25000),
        sampler=sampler,
        out_channels=2,
        add_boundary_transform=True,
        save_root="./output/experiment1/run1",
        check=False,
        device=0
    )

def train_actin_deepict_v4():
    """Train a network for actin segmentation on the deepict dataset.
    Same as v3, with ignore_label to mask background voxels from loss.
    """
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/")
    TRAIN_DIR = PARENT_DIR / "train"
    VAL_DIR = PARENT_DIR / "val"

    train_paths = [str(p) for p in TRAIN_DIR.glob("*.h5")]
    val_paths = [str(p) for p in VAL_DIR.glob("*.h5")]

    patch_shape = (64, 384, 384)
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-deepict-v4",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        n_iterations=int(25000),
        sampler=sampler,
        out_channels=2,
        add_boundary_transform=True,
        save_root="./output/experiment1/run3",
        ignore_label=-1,
        check=False,
        device=4
    )

def train_actin_optogenetics():
    """Train a network for actin segmentation on the EMPIAR-12292 dataset.
    """
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/h5")
   
    all_paths = [str(p) for p in PARENT_DIR.glob("*.h5")]
    train_paths = all_paths[:10]
    val_paths = all_paths[10:12]

    patch_shape = (64, 384, 384)
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-opto-v1",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        n_iterations=int(25000),
        sampler=sampler,
        out_channels=2,                                           
        add_boundary_transform=True,
        save_root="./output/experiment2/run1",
        check=False,
        device=1
    )

def main():
    train_actin_deepict_v4()
    #train_actin_optogenetics()


if __name__ == "__main__":
    main()
