"""This script contains an example for how to train a network for
a segmentation task with SynapseNet. This script covers the case of
supervised training, i.e. your data needs to contain annotations for
the structures you want to segment. If you want to use domain adaptation
to adapt an already trained network to your data without the need for
additional annotations then check out `domain_adaptation.py`.

You can download example data for this script from:
TODO zenodo link to Single-Ax / Chemical Fix data.
"""
import os
from glob import glob

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training import supervised_training


def main():
    # This is the folder that contains your training data.
    # The example was designed so that it runs for the sample data downloaded to
    # the folder './data'. If you want to train on your own data than change this filepath accordingily.

    # data_root_folder = "./data
    data_root_folder = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/data_summary/for_zenodo/vesicles/train"  # noqa

    # The training data should be saved as .h5 files, with:
    # an internal dataset called 'raw' that contains the image data
    # and another dataset that contains the training annotations.
    label_key = "labels/vesicles"

    # Get all files with ending .h5 in the training folder.
    files = sorted(glob(os.path.join(data_root_folder, "**", "*.h5"), recursive=True))

    # Crate a train / val split.
    train_ratio = 0.85
    train_paths, val_paths = train_test_split(files, test_size=1 - train_ratio, shuffle=True, random_state=42)

    # TODO explain
    # TODO 2d vs 3d training
    train_2d_model = True
    if train_2d_model:
        batch_size = 2
        model_name = "example-2d-vesicle-model"
        patch_shape = (1, 384, 384)
    else:
        batch_size = 1
        model_name = "example-3d-vesicle-model"
        patch_shape = (48, 256, 256)

    # TODO explain loader check
    check_loader = False

    # TODO explain the function and hint at advanced settings
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        val_paths=val_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples_train=None,
        n_samples_val=25,
        check=check_loader,
    )


if __name__ == "__main__":
    main()
