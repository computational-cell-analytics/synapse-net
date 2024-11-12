import os
from glob import glob

import numpy as np
import torch_em

from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from scipy.ndimage import binary_dilation, distance_transform_edt

from synaptic_reconstruction.training import supervised_training

TRAIN_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/compartments"
# TRAIN_ROOT = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/cooper/ground_truth/compartments/output/compartment_gt"  # noqa


def get_paths_2d():
    paths = sorted(glob(os.path.join(TRAIN_ROOT, "v1", "**", "*.h5"), recursive=True))
    return paths


def get_paths_3d():
    paths = sorted(glob(os.path.join(TRAIN_ROOT, "v2", "**", "*.h5"), recursive=True))
    paths += sorted(glob(os.path.join(TRAIN_ROOT, "v3", "**", "*.h5"), recursive=True))
    return paths


def label_transform_2d(seg):
    boundaries = find_boundaries(seg)
    distances = distance_transform_edt(~seg).astype("float32")
    distances /= distances.max()

    boundaries = gaussian(boundaries.astype("float32"), sigma=1.0)
    boundaries = rank.autolevel(img_as_ubyte(boundaries), disk(8)).astype("float") / 255

    distance_mask = seg != 0
    boundary_mask = binary_dilation(distance_mask, iterations=8)

    return np.stack([boundaries, distances, boundary_mask, distance_mask])


def label_transform_3d(seg):
    output = np.zeros((4,) + seg.shape, dtype="float32")
    for z in range(seg.shape[0]):
        out = label_transform_2d(seg[z])
        output[:, z] = out
    return output


def train_compartments_2d_v1():
    paths = get_paths_2d()
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)

    patch_shape = (384, 384)
    batch_size = 4

    check = False

    save_root = "."
    supervised_training(
        name="compartment_model_2d/v1",
        train_paths=train_paths,
        val_paths=val_paths,
        raw_key="data",
        label_key="/labels/compartments",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        save_root=save_root,
        label_transform=label_transform_2d,
        mask_channel=True,
        out_channels=1,
        n_samples_train=400,
        n_samples_val=40,
        n_iterations=int(2e4),
    )


def train_compartments_3d_v2():
    paths = get_paths_3d()
    train_paths, val_paths = train_test_split(paths, test_size=0.10, random_state=42)
    print("Number of train paths:", len(train_paths))
    print("Number of val paths:", len(val_paths))

    patch_shape = (64, 384, 384)
    batch_size = 1

    check = False
    sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=2)

    save_root = "."
    supervised_training(
        name="compartment_model_3d/v2",
        train_paths=train_paths,
        val_paths=val_paths,
        label_key="/labels/compartments",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        save_root=save_root,
        label_transform=label_transform_3d,
        mask_channel=True,
        n_samples_train=250,
        n_samples_val=25,
        n_iterations=int(5e4),
        out_channels=2,
        sampler=sampler,
        num_workers=8,
    )


def main():
    # train_compartments_2d_v1()
    train_compartments_3d_v2()


if __name__ == "__main__":
    main()
