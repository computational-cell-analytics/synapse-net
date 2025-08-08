import argparse
import h5py
import os
from pathlib import Path

from tqdm import tqdm
from elf.io import open_file

from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.inference.util import parse_tiling
from synapse_net.inference.inference import get_model_path

def _require_output_folders(output_folder):
    #seg_output = os.path.join(output_folder, "segmentations")
    seg_output = output_folder
    os.makedirs(seg_output, exist_ok=True)
    return seg_output

def get_volume(input_path):
    '''
    with h5py.File(input_path) as seg_file:
        input_volume = seg_file["raw"][:]
    '''
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

def run_vesicle_segmentation(input_path, output_path, mask_path, mask_key,tile_shape, halo, include_boundary, key_label, model_path=None, save_pred=False):
    tiling = parse_tiling(tile_shape, halo)
    print(f"using tiling {tiling}")
    input = get_volume(input_path)

    #check if we have a restricting mask for the segmentation
    if mask_path is not None:
        with open_file(mask_path, "r") as f:
                        mask = f[mask_key][:]
    else:
        mask = None
    
    if model_path is None:
        model_path = get_model_path("vesicles_3d")
        
    segmentation, prediction = segment_vesicles(input_volume=input, model_path=model_path, verbose=False, tiling=tiling, return_predictions=True, exclude_boundary=not include_boundary, mask = mask)
    foreground, boundaries = prediction[:2]

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

        key=f"predictions/{key_label}"
        if key in f:
            print("Skipping", input_path, "because", key, "exists")
        else:
            f.create_dataset(key, data=segmentation, compression="gzip")
        if save_pred:
            if f"prediction_{key_label}/foreground" in f and f"prediction_{key_label}/boundaries" in f:
                print(f"Skipping {input_path} because prediction_{key_label}/foreground and prediction_{key_label}/boundaries exist")
            else:
                f.create_dataset(f"prediction_{key_label}/foreground", data=foreground, compression="gzip")
                f.create_dataset(f"prediction_{key_label}/boundaries", data=boundaries, compression="gzip")

            
        if mask is not None:
            if mask_key in f:
                print("mask image already saved")
            else:
                f.create_dataset(mask_key, data = mask, compression = "gzip")
        
        


def segment_folder(args):
    input_files = []
    for root, dirs, files in os.walk(args.input_path):
        input_files.extend([
            os.path.join(root, name) for name in files if name.endswith(".h5")
        ])
    print(input_files)
    pbar = tqdm(input_files, desc="Run segmentation")
    for input_path in pbar:

        filename = os.path.basename(input_path)
        try:
            mask_path = os.path.join(args.mask_path, filename)
        except:
            print(f"Mask file not found for {input_path}")
            mask_path = None

        run_vesicle_segmentation(input_path, args.output_path, mask_path, args.mask_key, args.tile_shape, args.halo, args.include_boundary, args.key_label, args.model_path, args.save_pred)

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
        "--model_path", "-m", help="The filepath to the vesicle model."
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a h5 file with a mask that will be used to restrict the segmentation. Needs to be in combination with mask_key."
    )
    parser.add_argument(
        "--mask_key", help="Key name that holds the mask segmentation"
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
        "--include_boundary", action="store_true",
        help="Include vesicles that touch the top / bottom of the tomogram. By default these are excluded."
    )
    parser.add_argument(
        "--key_label", "-k", default = "vesicle_seg",
        help="Give the key name for saving the segmentation in h5."
    )
    parser.add_argument(
        "--save_pred", action="store_true",
        help="If set to true the prediction is also saved."
    )
    args = parser.parse_args()

    input_ = args.input_path
    
    if os.path.isdir(input_):
        segment_folder(args)
    else:
        run_vesicle_segmentation(input_, args.output_path, args.mask_path, args.mask_key, args.tile_shape, args.halo, args.include_boundary, args.key_label, args.model_path, args.save_pred)

    print("Finished segmenting!")

if __name__ == "__main__":
    main()