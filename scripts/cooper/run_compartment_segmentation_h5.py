import argparse
from functools import partial

from synapse_net.inference.compartments import segment_compartments
from synapse_net.inference.inference import get_model_path
from synapse_net.inference.util import inference_helper, parse_tiling

import h5py
import numpy as np
from elf.io import open_file

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

def run_compartment_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)

    if args.model is None:
        model_path = get_model_path("compartments")
    else:
        model_path = args.model

    # Call segment_compartments directly, since we need its outputs
    segmentation, predictions = segment_compartments(
        get_volume(args.input_path),
        model_path=model_path,
        verbose=True,
        tiling=tiling,
        scale=None,
        boundary_threshold=args.boundary_threshold,
        return_predictions=True
    )

    # Save outputs into input HDF5 file
    with h5py.File(args.input_path, "a") as f:
        pred_grp = f.require_group("predictions")

        if "comp_seg" in pred_grp:
            if args.force:
                del pred_grp["comp_seg"]
            else:
                raise RuntimeError("comp_seg already exists. Use --force to overwrite.")
        pred_grp.create_dataset("comp_seg", data=segmentation.astype(np.uint8), compression="gzip")

        if "boundaries" in pred_grp:
            if args.force:
                del pred_grp["boundaries"]
            else:
                raise RuntimeError("boundaries already exist. Use --force to overwrite.")
        pred_grp.create_dataset("boundaries", data=predictions.astype(np.float32), compression="gzip")

    print(f"Saved segmentation to: predictions/comp_seg")
    print(f"Saved boundaries to: predictions/boundaries")


def main():
    parser = argparse.ArgumentParser(description="Segment synaptic compartments in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the tomogram data."
    )
    parser.add_argument(
        "--model", "-m", help="The filepath to the compartment model."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present segmentation results."
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
        "--data_ext", default=".mrc", help="The extension of the tomogram data. By default .mrc."
    )
    parser.add_argument(
        "--boundary_threshold", type=float, default=0.4, help="Threshold that determines when the prediction of the network is foreground for the segmentation. Need higher threshold than default for TEM."
    )

    args = parser.parse_args()
    run_compartment_segmentation(args)


if __name__ == "__main__":
    main()
