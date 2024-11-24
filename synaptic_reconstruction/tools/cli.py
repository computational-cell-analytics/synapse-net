import argparse
from functools import partial

from .util import run_segmentation, get_model
from ..inference.util import inference_helper, parse_tiling


# TODO: handle kwargs
def segmentation_cli():
    parser = argparse.ArgumentParser(description="Run segmentation.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--model", "-m", required=True, help="The model type."
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a tif file with a mask that will be used to restrict the segmentation."
        "Can also be a directory with tifs if the filestructure matches input_path."
    )
    parser.add_argument("--input_key", "-k", required=False)
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

    model = get_model(args.model)
    tiling = parse_tiling(args.tile_shape, args.halo)

    segmentation_function = partial(
        run_segmentation, model=model, model_type=args.model, verbose=False, tiling=tiling,
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext,
    )
