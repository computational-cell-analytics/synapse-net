import os
from glob import glob

import h5py

from synapse_net.file_utils import read_mrc
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from synapse_net.tools.util import compute_scale_from_voxel_size
from synapse_net.inference.util import _Scaler


# Apply rescaling, depending on what the segmentation comparison shows.
def preprocess_training_data():
    root = "../data/tomograms"
    tomograms = glob(os.path.join(root, "**", "*.mrc"), recursive=True)
    tomograms += glob(os.path.join(root, "**", "*.rec"), recursive=True)
    tomograms = sorted(tomograms)

    train_folder = "./train_data"
    os.makedirs(train_folder, exist_ok=True)

    all_paths = []
    for i, tomo_path in enumerate(tomograms):
        out_path = os.path.join(train_folder, f"tomo{i}.h5")
        if os.path.exists(out_path):
            all_paths.append(out_path)
            continue

        data, voxel_size = read_mrc(tomo_path)
        scale = compute_scale_from_voxel_size(voxel_size, "ribbon")
        print("Scale factor:", scale)
        scaler = _Scaler(scale, verbose=True)
        data = scaler.scale_input(data)

        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=data, compression="gzip")
        all_paths.append(out_path)

    train_paths, val_paths = all_paths[:-1], all_paths[-1:]
    return train_paths, val_paths


def train_domain_adaptation(train_paths, val_paths):
    model_path = "/mnt/vast-nhr/home/pape41/u12086/inner-ear-da.pt"
    model_name = "adapted_otoferlin"

    patch_shape = [48, 384, 384]
    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        source_checkpoint=model_path,
        confidence_threshold=0.75,
        n_iterations=int(2.5*1e4),
    )


def main():
    train_paths, val_paths = preprocess_training_data()
    train_domain_adaptation(train_paths, val_paths)


if __name__ == "__main__":
    main()
