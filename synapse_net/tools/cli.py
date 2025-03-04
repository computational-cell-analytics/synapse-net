import argparse
from functools import partial

import torch
from ..imod.to_imod import export_helper, write_segmentation_to_imod_as_points, write_segmentation_to_imod
from ..inference.inference import _get_model_registry, get_model, get_model_training_resolution, run_segmentation
from ..inference.util import inference_helper, parse_tiling


def imod_point_cli():
    parser = argparse.ArgumentParser(
        description="Convert a vesicle segmentation to an IMOD point model, "
        "corresponding to a sphere for each vesicle in the segmentation."
    )
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k",
        help="The key in the segmentation files. If not given we assume that the segmentations are stored as tif."
        "If given, we assume they are stored as hdf5 files, and use the key to load the internal dataset."
    )
    parser.add_argument(
        "--min_radius", type=float, default=10.0,
        help="The minimum vesicle radius in nm. Objects that are smaller than this radius will be exclded from the export."  # noqa
    )
    parser.add_argument(
        "--radius_factor", type=float, default=1.0,
        help="A factor for scaling the sphere radius for the export. "
        "This can be used to fit the size of segmented vesicles to the best matching spheres.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present export results."
    )
    args = parser.parse_args()

    export_function = partial(
        write_segmentation_to_imod_as_points,
        min_radius=args.min_radius,
        radius_factor=args.radius_factor,
    )

    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=export_function,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


def imod_object_cli():
    parser = argparse.ArgumentParser(
        description="Convert segmented objects to close contour IMOD models."
    )
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", required=True,
        help="The filepath to the file or the directory containing the segmentations."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--segmentation_key", "-k",
        help="The key in the segmentation files. If not given we assume that the segmentations are stored as tif."
        "If given, we assume they are stored as hdf5 files, and use the key to load the internal dataset."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Whether to over-write already present export results."
    )
    args = parser.parse_args()
    export_helper(
        input_path=args.input_path,
        segmentation_path=args.segmentation_path,
        output_root=args.output_path,
        export_function=write_segmentation_to_imod,
        force=args.force,
        segmentation_key=args.segmentation_key,
    )


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
    model_names = list(_get_model_registry().urls.keys())
    model_names = ", ".join(model_names)
    parser.add_argument(
        "--model", "-m", required=True,
        help=f"The model type. The following models are currently available: {model_names}"
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
        help="The tile shape for prediction, in ZYX order. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction, in ZYX order. Increase the halo to minimize boundary artifacts."
    )
    parser.add_argument(
        "--data_ext", default=".mrc", help="The extension of the tomogram data. By default .mrc."
    )
    parser.add_argument(
        "--checkpoint", "-c", help="Path to a custom model, e.g. from domain adaptation.",
    )
    parser.add_argument(
        "--segmentation_key", "-s",
        help="If given, the outputs will be saved to an hdf5 file with this key. Otherwise they will be saved as tif.",
    )
    parser.add_argument(
        "--scale", type=float,
        help="The factor for rescaling the data before inference. "
        "By default, the scaling factor will be derived from the voxel size of the input data. "
        "If this parameter is given it will over-ride the default behavior. "
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Whether to print verbose information about the segmentation progress."
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        model = get_model(args.model)
    else:
        model = torch.load(args.checkpoint, weights_only=False)
        assert model is not None, f"The model from {args.checkpoint} could not be loaded."

    is_2d = "2d" in args.model
    tiling = parse_tiling(args.tile_shape, args.halo, is_2d=is_2d)

    # If the scale argument is not passed, then we get the average training resolution for the model.
    # The inputs will then be scaled to match this resolution based on the voxel size from the mrc files.
    if args.scale is None:
        model_resolution = get_model_training_resolution(args.model)
        model_resolution = tuple(model_resolution[ax] for ax in ("yx" if is_2d else "zyx"))
        scale = None
    # Otherwise, we set the model resolution to None and use the scaling factor provided by the user.
    else:
        model_resolution = None
        scale = (2 if is_2d else 3) * (args.scale,)

    segmentation_function = partial(
        run_segmentation, model=model, model_type=args.model, verbose=args.verbose, tiling=tiling,
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext,
        output_key=args.segmentation_key, model_resolution=model_resolution, scale=scale,
    )
