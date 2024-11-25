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
    # The example was designed so that it runs for the sample data downloaded to './data'.
    # If you want to train on your own data than change this filepath accordingly.
    # TODO update to match zenodo download
    data_root_folder = "./data/vesicles/train"

    # The training data should be saved as .h5 files, with:
    # an internal dataset called 'raw' that contains the image data
    # and another dataset that contains the training annotations.
    label_key = "labels/vesicles"

    # Get all files with the ending .h5 in the training folder.
    files = sorted(glob(os.path.join(data_root_folder, "**", "*.h5"), recursive=True))

    # Crate a train / val split.
    train_ratio = 0.85
    train_paths, val_paths = train_test_split(files, test_size=1 - train_ratio, shuffle=True, random_state=42)

    # We can either train a 2d or a 3d model. Whether a 2d or a 3d model is trained is derived from the patch shape.
    # If your training data for 2d is stored as images (i.e. 2d data) them choose a  patch shape of form Y x X,
    # e.g. (384, 384). If your data is stored in 3d, but you want to train a 2d model on it, choose a patch shape
    # of the form 1 x Y x X, e.g. (1, 384, 384).
    # If you want to train a 3d model then choose a patch shape of form Z x Y x X, e.g. (48, 256, 256).
    train_2d_model = True
    if train_2d_model:
        batch_size = 2  # You can increase the batch size if you have enough VRAM.
        # The model name determines the name of the checkpoint. E.g., for the name here the checkpoint will
        # be saved at: 'checkpoints/example-2d-vesicle-model/'.
        model_name = "example-2d-vesicle-model"
        # The patch shape for training. See futher explanations above.
        patch_shape = (1, 384, 384)
    else:
        batch_size = 1  # You can increase the batch size if you have enough VRAM.
        # See the explanations for model_name and patch_shape above.
        model_name = "example-3d-vesicle-model"
        patch_shape = (48, 256, 256)

    # If check_loader is set to True the training samples will be visualized via napari
    # instead of starting a training. This is useful to validate that the training data
    # is read correctly.
    check_loader = False

    # This function runs the training. Check out its documentation for
    # advanced settings to update the training procedure.
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
