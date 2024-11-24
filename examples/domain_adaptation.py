"""
"""

import os
from glob import glob

from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training import mean_teacher_adaptation
from synaptic_reconstruction.tools.util import get_model_path


def main():
    train_2d_model = True

    # TODO adjust to local paths
    data_root_folder_2d = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/data_summary/for_zenodo/2d_tem/train_unlabeled"  # noqa
    data_root_folder_3d = ""

    data_root_folder = data_root_folder_2d if train_2d_model else data_root_folder_3d

    # Get all files with ending .h5 in the training folder.
    files = sorted(glob(os.path.join(data_root_folder, "**", "*.h5"), recursive=True))

    # Crate a train / val split.
    train_ratio = 0.85
    train_paths, val_paths = train_test_split(files, test_size=1 - train_ratio, shuffle=True, random_state=42)

    if train_2d_model:
        model_name = "example-2d-adapted-model"
        patch_shape = (256, 256)
        batch_size = 4
        source_checkpoint = get_model_path(model_type="vesicles_2d")
    else:
        model_name = "example-3d-adapted-model"
        patch_shape = (48, 256, 256)
        batch_size = 1
        source_checkpoint = get_model_path(model_type="vesicles_3d")

    n_iterations = int(2.5e4)
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
