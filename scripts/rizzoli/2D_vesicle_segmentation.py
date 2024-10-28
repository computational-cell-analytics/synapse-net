import argparse
import h5py
import os
from pathlib import Path

from tqdm import tqdm
import torch
import torch_em
import numpy as np

from synaptic_reconstruction.inference.vesicles import segment_vesicles
from synaptic_reconstruction.inference.util import parse_tiling

def get_2D_tiling():
    """Determine the tile shape and halo depending on the available VRAM.
    """
    if torch.cuda.is_available():
        print("Determining suitable tiling")

        # We always use the same default halo.
        halo = {"x": 240, "y": 240, "z": 0}

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        #TODO test tiling if it works
        if vram >= 80:
            tile = {"x": 1520, "y": 1520, "z": 1}
        elif vram >= 40:
            tile = {"x": 1360, "y": 1360, "z": 1}
        elif vram >= 20:
            tile = {"x": 1200, "y": 1200, "z": 1}
        else:
            # TODO determine tilings for smaller VRAM
            raise NotImplementedError

        print(f"using tile size: {tile}")
        tiling = {"tile": tile, "halo": halo}

    else:
        print("Using default tiling")
        tiling = {
            "tile": {"x": 1200, "y": 1200, "z": 1},
            "halo": {"x": 240, "y": 240, "z": 0},
        }

    return tiling

def _require_output_folders(output_folder):
    #seg_output = os.path.join(output_folder, "segmentations")
    seg_output = output_folder
    os.makedirs(seg_output, exist_ok=True)
    return seg_output

def get_volume(input_path):
    with h5py.File(input_path) as seg_file:
        input_volume = seg_file["raw"][:]
    return input_volume

def run_vesicle_segmentation(input_path, output_path, model_path, tile_shape, halo, include_boundary, key_label, scale):

    tiling = get_2D_tiling()

    if tile_shape is None:
        tile_shape = (tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"])
    if halo is None:
        halo = (tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"])

    tiling = parse_tiling(tile_shape, halo)
    input = get_volume(input_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch_em.util.load_model(checkpoint=model_path, device=device)

    def process_slices(input_volume, scale):
        processed_slices = []
        foreground = []
        boundaries = []
        for z in range(input_volume.shape[0]):
            slice_ = input_volume[z, :, :]
            segmented_slice, prediction_slice = segment_vesicles(input_volume=slice_, model=model, verbose=False, tiling=tiling, return_predictions=True, scale = scale, exclude_boundary=not include_boundary)
            processed_slices.append(segmented_slice)
            foreground_pred_slice, boundaries_pred_slice = prediction_slice[:2]
            foreground.append(foreground_pred_slice)
            boundaries.append(boundaries_pred_slice)
        return processed_slices, foreground, boundaries

    if input.ndim == 2:
        segmentation, prediction = segment_vesicles(input_volume=input, model=model, verbose=False, tiling=tiling, return_predictions=True, scale = scale, exclude_boundary=not include_boundary)
        foreground, boundaries = prediction[:2]
    else:
        segmentation, foreground, boundaries = process_slices(input, scale)

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

        key=f"vesicles/segment_from_{key_label}"
        if key in f:
            print("Skipping", input_path, "because", key, "exists")
        else:
            f.create_dataset(key, data=segmentation, compression="gzip")
            f.create_dataset(f"prediction_{key_label}/foreground", data = foreground, compression="gzip")
            f.create_dataset(f"prediction_{key_label}/boundaries", data = boundaries, compression="gzip")
        
        


def segment_folder(args):
    input_files = []
    for root, dirs, files in os.walk(args.input_path):
        input_files.extend([
            os.path.join(root, name) for name in files if name.endswith(".h5")
        ])
    print(input_files)
    pbar = tqdm(input_files, desc="Run segmentation")
    for input_path in pbar:
        run_vesicle_segmentation(input_path, args.output_path, args.model_path, args.tile_shape, args.halo, args.include_boundary, args.key_label, args.scale)

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
        "--model_path", "-m", required=True, help="The DIRECTORY path to the vesicle model."
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
        "--key_label", "-k", default = "combined_vesicles",
        help="Give the key name for saving the segmentation in h5."
    )
    parser.add_argument(
        "--scale", "-s", type=float, nargs=2,
        help="Scales the input data."
    )
    args = parser.parse_args()

    input_ = args.input_path
    
    if os.path.isdir(input_):
        segment_folder(args)
    else:
        run_vesicle_segmentation(input_, args.output_path, args.model_path, args.tile_shape, args.halo, args.include_boundary, args.key_label, args.scale)

    print("Finished segmenting!")

if __name__ == "__main__":
    main()