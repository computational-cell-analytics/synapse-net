"""This script contains an example for using domain adptation to
transfer a trained model for vesicle segmentation to a new dataset from a different data distribution,
e.g. data from regular transmission electron microscopy (2D) instead of electron tomography or data from
a different electron tomogram with different specimen and sample preparation.
You don't need any annotations in the new domain to run this script.

You can download example data for this script from:
- Adaptation to 2d TEM data: TODO zenodo link
- Adaptation to different tomography data: TODO zenodo link
"""

import os
from glob import glob

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training import mean_teacher_adaptation
from synaptic_reconstruction.tools.util import get_model_path


def main():
    # Choose whether to adapt the model to 2D or to 3D data.
    train_2d_model = True

    # TODO adjust to zenodo downloads
    # These are the data folders for the example data downloaded from zenodo.
    # Update these paths to apply the script to your own data.
    # Check out the example data to see the data format for training.
    data_root_folder_2d = "./data/2d_tem/train_unlabeled"
    data_root_folder_3d = "./data/..."

    # Choose the correct data folder depending on 2d/3d training.
    data_root_folder = data_root_folder_2d if train_2d_model else data_root_folder_3d

    # Get all files with ending .h5 in the training folder.
    files = sorted(glob(os.path.join(data_root_folder, "**", "*.h5"), recursive=True))

    # Crate a train / val split.
    train_ratio = 0.85
    train_paths, val_paths = train_test_split(files, test_size=1 - train_ratio, shuffle=True, random_state=42)

    # Choose settings for the 2d or 3d domain adaptation.
    if train_2d_model:
        # This is the name of the checkpoint of the adapted model.
        # For the name here the checkpoint will be stored in './checkpoints/example-2d-adapted-model'
        model_name = "example-2d-adapted-model"
        # The training patch size.
        patch_shape = (256, 256)
        # The batch size for training. You can increase this if you have enough VRAM.
        batch_size = 4
        # Get the checkpoint of the pretrained model for 2d vesicle segmentation.
        source_checkpoint = get_model_path(model_type="vesicles_2d")
    else:
        # This is the name of the checkpoint of the adapted model.
        # For the name here the checkpoint will be stored in './checkpoints/example-3d-adapted-model'
        model_name = "example-3d-adapted-model"
        # The training patch size.
        patch_shape = (48, 256, 256)
        # The batch size for training. You can increase this if you have enough VRAM.
        batch_size = 1
        # Get the checkpoint of the pretrained model for d vesicle segmentation.
        source_checkpoint = get_model_path(model_type="vesicles_3d")

    # We set the number of training iterations to 25,000.
    n_iterations = int(2.5e4)

    # This function runs the domain adaptation. Check out its documentation for
    # advanced settings to update the training procedure.
    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        source_checkpoint=source_checkpoint,
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_iterations=n_iterations,
        confidence_threshold=0.75,
    )


if __name__ == "__main__":
    main()
