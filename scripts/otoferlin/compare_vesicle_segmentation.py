import os

import h5py

from skimage.exposure import equalize_adapthist
from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.file_utils import read_mrc
from synapse_net.tools.util import get_model, compute_scale_from_voxel_size, load_custom_model
from tqdm import tqdm

from common import get_all_tomograms, get_seg_path


def compare_vesicles(tomo_path):
    seg_path = get_seg_path(tomo_path)
    seg_folder = os.path.split(seg_path)[0]
    os.makedirs(seg_folder, exist_ok=True)

    model_paths = {
        "adapted_v1": "/mnt/vast-nhr/home/pape41/u12086/inner-ear-da.pt",
        "adapted_v2": "./domain_adaptation/checkpoints/otoferlin_da.pt"
    }
    for model_type in ("vesicles_3d", "adapted_v1", "adapted_v2"):
        for use_clahe in (False, True):
            seg_key = f"vesicles/{model_type}"
            if use_clahe:
                seg_key += "_clahe"

            if os.path.exists(seg_path):
                with h5py.File(seg_path, "r") as f:
                    if seg_key in f:
                        continue

            tomogram, voxel_size = read_mrc(tomo_path)
            if use_clahe:
                tomogram = equalize_adapthist(tomogram, clip_limit=0.03)

            if model_type == "vesicles_3d":
                model = get_model(model_type)
                scale = compute_scale_from_voxel_size(voxel_size, model_type)
            else:
                model_path = model_paths[model_type]
                model = load_custom_model(model_path)
                scale = compute_scale_from_voxel_size(voxel_size, "ribbon")

            seg = segment_vesicles(tomogram, model=model, scale=scale)
            with h5py.File(seg_path, "a") as f:
                f.create_dataset(seg_key, data=seg, compression="gzip")


def main():
    tomograms = get_all_tomograms()
    for tomo in tqdm(tomograms):
        compare_vesicles(tomo)


if __name__ == "__main__":
    main()
