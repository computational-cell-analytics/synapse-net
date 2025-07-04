import argparse
import os
from glob import glob

import h5py
from synapse_net.inference.active_zone import segment_active_zone
from torch_em.util import load_model
from tqdm import tqdm

from common import get_file_names, get_split_folder, ALL_NAMES, INPUT_ROOT, OUTPUT_ROOT


def run_prediction(model, name, split_folder, version, split_names, in_path):
    if in_path:
        file_paths = glob(os.path.join(in_path, name, "*.h5"))
        file_names = [os.path.basename(path) for path in file_paths]
    else:
        file_names = get_file_names(name, split_folder, split_names=split_names)

    output_folder = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(output_folder, exist_ok=True)
    output_key = f"predictions/az/v{version}"
    output_key_seg = f"predictions/az/seg_v{version}"

    for fname in tqdm(file_names):
        if in_path:
            input_path=os.path.join(in_path, name, fname)
        else:
            input_path = os.path.join(INPUT_ROOT, name, fname)
        print(f"segmenting {input_path}")

        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if output_key in f and output_key_seg in f:
                    print(f"skipping, because {output_key} and {output_key_seg} already exists in {output_path}")
                    continue

        with h5py.File(input_path, "r") as f:
            raw = f["raw"][:]

        seg, pred = segment_active_zone(raw, model=model, verbose=False, return_predictions=True)
        with h5py.File(output_path, "a") as f:
            if output_key in f:
                print(f"{output_key} already saved")
            else:
                f.create_dataset(output_key, data=pred, compression="lzf")
            if output_key_seg in f:
                print(f"{output_key_seg} already saved")
            else:
                f.create_dataset(output_key_seg, data=seg, compression="lzf")
                


def get_model(version):
    assert version in (3, 4, 5, 6, 7)
    split_folder = get_split_folder(version)
    if version == 3:
        model_path = os.path.join(split_folder, "checkpoints", "3D-AZ-model-TEM_STEM_ChemFix_wichmann-v3")
    elif version ==6:
        model_path = "/mnt/ceph-hdd/cold/nim00007/models/AZ/v6/"
    elif version == 7:
        model_path = "/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models/ConstantinAZ/checkpoints/v7/"
    else:
        model_path = os.path.join(split_folder, "checkpoints", f"v{version}")
    model = load_model(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", type=int)
    parser.add_argument("--names", nargs="+", default=ALL_NAMES)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input", "-i", default=None)

    args = parser.parse_args()

    if args.model_path:
        model = load_model(model_path)
    else:
        model = get_model(args.version)

    split_folder = get_split_folder(args.version)

    for name in args.names:
        run_prediction(model, name, split_folder, args.version, args.splits, args.input)
    
    print("Finished segmenting!")


if __name__ == "__main__":
    main()
