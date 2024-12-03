import argparse
import h5py
import os
from pathlib import Path

from tqdm import tqdm
from elf.io import open_file

from synaptic_reconstruction.inference.compartments import segment_compartments
from synaptic_reconstruction.inference.util import parse_tiling

def _require_output_folders(output_folder):
    #seg_output = os.path.join(output_folder, "segmentations")
    seg_output = output_folder
    os.makedirs(seg_output, exist_ok=True)
    return seg_output

def get_volume(input_path):

    with open_file(input_path, "r") as f:

        # Try to automatically derive the key with the raw data.
        keys = list(f.keys())
        if len(keys) == 1:
            key = keys[0]
        elif "data" in keys:
            key = "data"
        elif "raw" in keys:
            key = "raw"

        input_volume = f[key][:]
    return input_volume

def run_compartment_segmentation(input_path, output_path, model_path, tile_shape, halo, key_label):
    tiling = parse_tiling(tile_shape, halo)
    print(f"using tiling {tiling}")
    input = get_volume(input_path)

    segmentation, prediction = segment_compartments(input_volume=input, model_path=model_path, verbose=False, tiling=tiling, return_predictions=True, scale=[0.25, 0.25, 0.25],boundary_threshold=0.2, postprocess_segments=False)

    seg_output = _require_output_folders(output_path)
    file_name = Path(input_path).stem
    seg_path = os.path.join(seg_output, f"{file_name}.h5")

    #check
    os.makedirs(Path(seg_path).parent, exist_ok=True)

    print(f"Saving results in {seg_path}")
    with h5py.File(seg_path, "a") as f:
        if "raw" in f:
            print("raw image already saved")
        else:
            f.create_dataset("raw", data=input, compression="gzip")

        key=f"compartments/segment_from_{key_label}"
        if key in f:
            print("Skipping", input_path, "because", key, "exists")
        else:
            f.create_dataset(key, data=segmentation, compression="gzip")
            f.create_dataset(f"compartment_pred_{key_label}/foreground", data = prediction, compression="gzip")
        
        


def segment_folder(args):
    input_files = []
    for root, dirs, files in os.walk(args.input_path):
        input_files.extend([
            os.path.join(root, name) for name in files if name.endswith(args.data_ext)
        ])
    print(input_files)
    pbar = tqdm(input_files, desc="Run segmentation")
    for input_path in pbar:
        run_compartment_segmentation(input_path, args.output_path, args.model_path, args.tile_shape, args.halo, args.key_label)

def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the vesicle model."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3,
        help="The tile shape for prediction. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction. Increase the halo to minimize boundary artifacts."
    )
    parser.add_argument(
        "--data_ext", "-d", default=".h5", help="The extension of the tomogram data. By default .h5."
    )
    parser.add_argument(
        "--key_label", "-k", default = "3Dmodel_v1",
        help="Give the key name for saving the segmentation in h5."
    )
    args = parser.parse_args()

    input_ = args.input_path
    
    if os.path.isdir(input_):
        segment_folder(args)
    else:
        run_compartment_segmentation(input_, args.output_path, args.model_path, args.tile_shape, args.halo, args.key_label)

    print("Finished segmenting!")

if __name__ == "__main__":
    main()