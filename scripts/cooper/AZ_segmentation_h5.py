import argparse
import h5py
import os
import json
from pathlib import Path

from tqdm import tqdm
from elf.io import open_file

from synaptic_reconstruction.inference.AZ import segment_AZ
from synaptic_reconstruction.inference.util import parse_tiling

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

def run_AZ_segmentation(input_path, output_path, model_path, mask_path, mask_key,tile_shape, halo, key_label, compartment_seg):
    tiling = parse_tiling(tile_shape, halo)
    print(f"using tiling {tiling}")
    input = get_volume(input_path)

    #check if we have a restricting mask for the segmentation
    if mask_path is not None:
        with open_file(mask_path, "r") as f:
                        mask = f[mask_key][:]
    else:
        mask = None

    #check if intersection with compartment is necessary
    if compartment_seg is None:
        foreground = segment_AZ(input_volume=input, model_path=model_path, verbose=False, tiling=tiling, mask = mask)
        intersection = None
    else:
        with open_file(compartment_seg, "r") as f:
            compartment = f["/labels/compartment"][:]
        foreground, intersection = segment_AZ(input_volume=input, model_path=model_path, verbose=False, tiling=tiling, mask = mask, compartment=compartment)

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

        key=f"AZ/segment_from_{key_label}"
        if key in f:
            print("Skipping", input_path, "because", key, "exists")
        else:
            f.create_dataset(key, data=foreground, compression="gzip")

        if mask is not None:
            if mask_key in f:
                print("mask image already saved")
            else:
                f.create_dataset(mask_key, data = mask, compression = "gzip")

        if intersection is not None:
            intersection_key = "AZ/compartment_AZ_intersection"
            if intersection_key in f:
                print("intersection already saved")
            else:
                f.create_dataset(intersection_key, data = intersection, compression = "gzip")
        
        


def segment_folder(args, valid_files):
    input_files = [os.path.join(root, name) for root, _, files in os.walk(args.input_path) for name in files if name.endswith(args.data_ext)]
    input_files = [f for f in input_files if f in valid_files] if valid_files else input_files
    print(input_files)
    
    pbar = tqdm(input_files, desc="Run segmentation")
    for input_path in pbar:

        filename = os.path.basename(input_path)
        try:
            mask_path = os.path.join(args.mask_path, filename)
        except:
            print(f"Mask file not found for {input_path}")
            mask_path = None
        
        if args.compartment_seg is not None:
            try:
                compartment_seg = os.path.join(args.compartment_seg, os.path.splitext(filename)[0] + '.h5')
            except:
                print(f"compartment file not found for {input_path}")
                compartment_seg = None
        else:
            compartment_seg = None

        run_AZ_segmentation(input_path, args.output_path, args.model_path, mask_path, args.mask_key, args.tile_shape, args.halo, args.key_label, compartment_seg)

def get_dataset(json_file, input_path, sets=["test"]):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {os.path.join(input_path, f) for dataset in sets for f in data.get(dataset, [])}


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
        "--json_path", "-j",  help="The filepath to the json file that stores the train, val, and test split."
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
        "--key_label", "-k", default = "combined_vesicles",
        help="Give the key name for saving the segmentation in h5."
    )
    parser.add_argument(
        "--data_ext", "-d", default = ".h5",
        help="Format extension of data to be segmented, default is .h5."
    )
    parser.add_argument(
        "--compartment_seg", "-c", default = None,
        help="Path to compartment segmentation."
        "If the compartment segmentation was executed before, this will add a key to output file that stores the intersection between compartment boundary and AZ."
        "Maybe need to adjust the compartment key that the segmentation is stored under"
    )
    args = parser.parse_args()

    input_ = args.input_path
    valid_files = get_dataset(args.json_path, input_) if args.json_path else None
    
    if valid_files:
        if len(valid_files) == 1:
            run_AZ_segmentation(next(iter(valid_files)), args.output_path, args.model_path, args.mask_path, args.mask_key, args.tile_shape, args.halo, args.key_label, args.compartment_seg)
        else:
            segment_folder(args, valid_files)
    elif os.path.isdir(args.input_path):
        segment_folder(args, valid_files)
    else:
        run_AZ_segmentation(args.input_path, args.output_path, args.model_path, args.mask_path, args.mask_key, args.tile_shape, args.halo, args.key_label, args.compartment_seg)
    
    print("Finished segmenting!")

if __name__ == "__main__":
    main()