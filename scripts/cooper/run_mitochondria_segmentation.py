import argparse
from functools import partial

from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.inference import get_model_path
from synapse_net.inference.util import inference_helper, parse_tiling


def run_mitochondria_segmentation(args):
    if args.model is None:
        model_path = get_model_path("mitochondria")
    else:
        model_path = args.model

    tiling = parse_tiling(args.tile_shape, args.halo)
    segmentation_function = partial(
        segment_mitochondria, model_path=model_path, verbose=False, tiling=tiling, scale=[0.5, 0.5, 0.5]
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext
    )


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentation will be saved."
    )
    parser.add_argument(
        "--model", "-m", help="The filepath to the mitochondria model."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present segmentation results."
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a tif file with a mask that will be used to restrict the segmentation."
        "Can also be a directory with tifs if the filestructure matches input_path."
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
    run_mitochondria_segmentation(args)


if __name__ == "__main__":
    main()
