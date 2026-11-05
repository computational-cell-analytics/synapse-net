"""Segment mitochondria in volume EM data with SynapseNet's volume EM mitochondria model.

The model is not part of the model registry, so its checkpoint has to be passed explicitly via '-m'.
See `synapse_net.training.mitochondria_vol_em` for how it was trained.

The preprocessing is part of the model and is reproduced here in two steps:

- The white filler borders are removed from the **whole volume** before prediction. This cannot be
  done in the per-block 'preprocess' of 'segment_mitochondria', because 'get_prediction' standardizes
  a numpy input volume before it runs that, and the filler is identified by its literal value of 255,
  which no longer exists after standardizing.
- The blocks are then percentile-normalized via 'preprocess', exactly as during training. The
  percentile normalization is invariant under the standardization that 'get_prediction' applies, so
  the network sees the same values either way.

Pass '--no_white_patch_fix' to skip the filler removal, which is only correct for data that was not
cut out of a larger volume. On a block that is 14% filler, leaving it in cost 0.04 Dice (0.79 instead
of 0.84), because the filler skews the percentile normalization of every tile it overlaps.

The data is segmented at its native resolution by default, because the model is meant to be applied to
data at its own training resolution of 25 nm in z and 5 nm in xy. Pass '--scale' for data at a
different voxel size; it cannot be derived automatically, because the voxel size is only read for mrc
files and the volume EM data is stored as hdf5.
"""

import argparse
from functools import partial

import torch_em

from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.util import inference_helper, parse_tiling
from synapse_net.training.transform import remove_white_patches

# The post-processing defaults were found by a grid search on the 4007 dataset. Note that the grid
# search used a different (out-of-core) watershed implementation, so these are a starting point for
# the watershed of 'segment_mitochondria' rather than a tuned optimum for it.
SEED_DISTANCE = 1
BOUNDARY_THRESHOLD = 0.12
AREA_THRESHOLD = 200
MIN_SIZE = 1000


def segment_mitochondria_vol_em(input_volume, white_patch_min_size=20, **kwargs):
    """Remove the white filler borders of a volume EM block and then segment its mitochondria.

    Args:
        input_volume: The volume to segment, before any normalization.
        white_patch_min_size: The minimal size (in voxels) of a filler component that is removed.
            Pass None to segment without removing the filler.
        kwargs: Additional keyword arguments for `synapse_net.inference.mitochondria.segment_mitochondria`.

    Returns:
        The mitochondria segmentation.
    """
    if white_patch_min_size is not None:
        # This has to happen before the rescaling inside of 'segment_mitochondria', which would
        # interpolate the filler away from its literal value.
        input_volume = remove_white_patches(input_volume, min_size=white_patch_min_size)
    return segment_mitochondria(input_volume, **kwargs)


def run_mitochondria_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)
    segmentation_function = partial(
        segment_mitochondria_vol_em,
        white_patch_min_size=None if args.no_white_patch_fix else args.white_patch_min_size,
        model_path=args.model,
        tiling=tiling,
        preprocess=torch_em.transform.raw.normalize_percentile,
        min_size=args.min_size,
        seed_distance=args.seed_distance,
        boundary_threshold=args.boundary_threshold,
        area_threshold=args.area_threshold,
        ws_block_shape=tuple(args.ws_block_shape),
        ws_halo=tuple(args.ws_halo),
        verbose=args.verbose,
    )
    inference_helper(
        args.input_path, args.output_path, segmentation_function,
        mask_input_path=args.mask_path, force=args.force, data_ext=args.data_ext,
        output_key=args.segmentation_key, scale=args.scale,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the hdf5 file or the directory containing the volume EM data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to the directory where the segmentation will be saved."
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="The filepath to the volume EM mitochondria model. This model is not part of the model registry."
    )
    parser.add_argument(
        "--segmentation_key", "-s", default="seg",
        help="The key the segmentation is saved under. Pass an empty string to save tif files instead."
    )
    parser.add_argument(
        "--data_ext", default=".h5", help="The extension of the volume EM data. By default .h5."
    )
    parser.add_argument(
        "--mask_path", help="The filepath to a tif file with a mask that will be used to restrict the segmentation."
        "Can also be a directory with tifs if the filestructure matches input_path."
    )
    parser.add_argument(
        "--force", action="store_true", help="Whether to over-write already present segmentation results."
    )
    parser.add_argument(
        "--scale", type=float, nargs=3,
        help="The factor for rescaling the data before prediction, in ZYX. By default the data is segmented at its "
        "native resolution, which is correct for data at the training voxel size of 25 x 5 x 5 nm."
    )
    parser.add_argument(
        "--no_white_patch_fix", action="store_true",
        help="Do not remove the white filler borders before normalizing. Only correct for data that was not cut out "
        "of a larger volume."
    )
    parser.add_argument(
        "--white_patch_min_size", type=int, default=20,
        help="The minimal size (in voxels) of a white filler component that is removed."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3, help="The tile shape for prediction, in ZYX. Lower it if GPU memory is "
        "insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3, help="The halo for prediction, in ZYX. Increase it to minimize boundary artifacts."
    )
    parser.add_argument(
        "--min_size", type=int, default=MIN_SIZE, help="The minimum size of a mitochondrion, in voxels."
    )
    parser.add_argument(
        "--seed_distance", type=int, default=SEED_DISTANCE, help="The distance threshold for the seeded watershed."
    )
    parser.add_argument(
        "--boundary_threshold", type=float, default=BOUNDARY_THRESHOLD,
        help="The boundary threshold for the distance calculation."
    )
    parser.add_argument(
        "--area_threshold", type=int, default=AREA_THRESHOLD,
        help="The maximum area (in pixels) of holes that are filled in the segmentation."
    )
    parser.add_argument(
        "--ws_block_shape", type=int, nargs=3, default=[128, 256, 256],
        help="The block shape for the seeded watershed, in ZYX."
    )
    parser.add_argument(
        "--ws_halo", type=int, nargs=3, default=[48, 48, 48], help="The halo for the seeded watershed, in ZYX."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Whether to print information about the segmentation progress."
    )

    args = parser.parse_args()
    if not args.segmentation_key:
        args.segmentation_key = None
    run_mitochondria_segmentation(args)


if __name__ == "__main__":
    main()
