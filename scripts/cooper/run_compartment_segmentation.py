import argparse
from functools import partial

from synapse_net.inference.compartments import segment_compartments
from synapse_net.inference.inference import get_model_path
from synapse_net.inference.util import inference_helper, parse_tiling


def run_compartment_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)

    if args.model is None:
        model_path = get_model_path("compartments")
    else:
        model_path = args.model

    segmentation_function = partial(
        segment_compartments, model_path=model_path, verbose=False, tiling=tiling, scale=[0.25, 0.25, 0.25]
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function, force=args.force, data_ext=args.data_ext
    )


def main():
    parser = argparse.ArgumentParser(description="Segment synaptic compartments in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentation will be saved."
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

    args = parser.parse_args()
    run_compartment_segmentation(args)


if __name__ == "__main__":
    main()
