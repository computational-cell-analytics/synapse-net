"""Segment volume EM blocks with micro-sam at a chosen voxel size.

This is the micro-sam baseline for compare_models.py, the counterpart of segment_mitonet.py. Each
block is resampled to the target voxel size, segmented with micro-sam's automatic instance
segmentation (AIS: the instance segmentation decoder of an EM organelle model), and the result is
resized back to the native grid with nearest-neighbor interpolation. The ground truth is never
resampled; compare_models.py scores the native 'seg' against it with '-s', and
score_voxel_size_sweep.py scores a whole voxel-size sweep.

micro-sam segments the xy slices in 2D and merges them along z; it has no ortho-plane mode, so there
is one mode, 'xy', written to '<prefix>-xy'. The prefix names the setting like segment_mitonet.py
does, e.g. 'microsam-iso12.5' or 'microsam-z25-xy10'.

The encoder scale is controlled, not only the resampling. SAM resizes every image, and every tile,
so that its longest side is 1024 px, up or down. Passed as-is, a slice of these 8 um blocks would
reach the encoder at ~7.8 nm/px at every target voxel size, only more or less blurred. So every
image or tile that holds data is made exactly 1024 px, and SAM runs at scale 1:
  a slice of at most 1024 px  is reflect-padded to 1024 x 1024 and segmented untiled
  a larger slice              is reflect-padded onto a canvas on which micro-sam's tiling, with
                              tiles of 768 and a halo of 128, gives every tile holding data its full
                              1024 px outer block; micro-sam clips the halo at the canvas border,
                              so the data starts one tile in
The padding is cropped away after the segmentation.

The size filter is matched to synapse-net in physical volume, as for MitoNet
(voxel_sweep.min_size_for). Everything else is micro-sam's: the AIS defaults, and the 3D merge with
gap_closing 2 and min_z_extent 2, the defaults of its napari annotator for automatic 3D segmentation.

This runs in the micro-sam environment, which does not have synapse_net, so it does not import
compare_models.py; the block lookup and resampling come from voxel_sweep.py.

Usage:
    python segment_microsam.py [--voxel_size Z Y X] [--model_type vit_b_em_organelles] [--force]
    python segment_microsam.py --split_file SPLIT.json --split_key val --segmentation_root ROOT ...
"""

import argparse
import importlib.metadata
import math
import os
import sys

import h5py
import numpy as np
from skimage.segmentation import relabel_sequential

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel_sweep import (  # noqa: E402
    NATIVE_VOXEL_SIZE, TEST_SPLIT, find_split_blocks, find_test_blocks, min_size_for, n_objects,
    setting_prefix, to_native_resolution, to_target_resolution,
)

SEGMENTATION_ROOT = "/mnt/lustre-grete/usr/u15205/volume-em/repro-comparison"
MODEL_TYPE = "vit_b_em_organelles"

# The input size of the SAM image encoder, and a tiling whose tiles with halo have exactly that size.
SAM_INPUT = 1024
TILE, HALO = 768, 128
assert TILE + 2 * HALO == SAM_INPUT

# The defaults of InstanceSegmentationWithDecoder.generate, set explicitly so the settings recorded
# with the segmentation stay true if a newer micro-sam changes them.
AIS_PARAMS = dict(
    center_distance_threshold=0.5, boundary_distance_threshold=0.5, foreground_threshold=0.5,
    foreground_smoothing=1.0, distance_smoothing=1.6, min_size=0,
)
# The defaults of micro-sam's napari annotator for automatic 3D segmentation; the merge along z
# itself is a multicut with beta 0.5, fixed inside automatic_3d_segmentation.
GAP_CLOSING = 2
MIN_Z_EXTENT = 2


def _axis_layout(n, tiled):
    """The padding before and after one axis of n px, so that SAM sees the data at scale 1."""
    if not tiled:
        before = (SAM_INPUT - n) // 2
        return before, SAM_INPUT - n - before
    # micro-sam's blocking starts at 0 and clips the halo at the border, so the first tile never has
    # its full outer block. The data starts at TILE, which leaves that tile to the padding, and the
    # canvas ends one halo after the last tile holding data.
    n_data_tiles = math.ceil(n / TILE)
    total = (n_data_tiles + 1) * TILE + HALO
    return TILE, total - TILE - n


def to_sam_canvas(volume):
    """Reflect-pad a volume in y and x so that every image or tile holding data is SAM_INPUT px.

    Returns the canvas, the slice that crops the data back out of it, and the tile shape and halo
    for micro-sam, which are None for an untiled canvas.
    """
    tiled = max(volume.shape[1:]) > SAM_INPUT
    pads = [(0, 0)] + [_axis_layout(n, tiled) for n in volume.shape[1:]]
    canvas = np.pad(volume, pads, mode="reflect")
    crop = tuple(slice(before, before + n) for (before, _), n in zip(pads, volume.shape))
    if tiled:
        return canvas, crop, (TILE, TILE), (HALO, HALO)
    return canvas, crop, None, None


def filter_small_objects(seg, min_size):
    """Drop the objects below a size and relabel the rest consecutively."""
    ids, counts = np.unique(seg, return_counts=True)
    too_small = ids[(counts < min_size) & (ids != 0)]
    if too_small.size:
        seg = seg.copy()
        seg[np.isin(seg, too_small)] = 0
    seg, _, _ = relabel_sequential(seg)
    return seg.astype(np.uint32)


def segment_microsam(predictor, segmenter, volume, min_size):
    """Segment a volume with micro-sam's AIS at scale 1 and return the segmentation of the data."""
    from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation

    canvas, crop, tile_shape, halo = to_sam_canvas(volume)
    seg = automatic_3d_segmentation(
        canvas, predictor, segmenter, gap_closing=GAP_CLOSING, min_z_extent=MIN_Z_EXTENT,
        tile_shape=tile_shape, halo=halo, verbose=False, **AIS_PARAMS,
    )
    return filter_small_objects(seg[crop], min_size), canvas.shape, tile_shape, halo


def settings(model_type, voxel_size, invert, min_size, inference_shape, native_shape, canvas_shape,
             tile_shape, halo, n_lost, source):
    """The settings a segmentation was made with, stored as the attributes of its 'seg'."""
    effective = [n * size / m for n, size, m in zip(native_shape, NATIVE_VOXEL_SIZE, inference_shape)]
    return {
        "model": model_type,
        "mode": "xy slices merged in z",
        "segmentation_mode": "ais",
        "target_voxel_size_nm": np.array(voxel_size, dtype=float),
        "inference_voxel_size_nm": np.round(effective, 2),
        "invert": bool(invert),
        # AIS has a 'min_size' of its own, per slice; the 3D size filter keeps the plain name.
        **{("ais_min_size" if key == "min_size" else key): value for key, value in AIS_PARAMS.items()},
        "min_size": min_size,
        "gap_closing": GAP_CLOSING,
        "min_z_extent": MIN_Z_EXTENT,
        "sam_input": SAM_INPUT,
        "canvas_shape": np.array(canvas_shape),
        "tile_shape": np.array(tile_shape) if tile_shape else "none",
        "halo": np.array(halo) if halo else "none",
        "n_lost_to_native": n_lost,
        "blocks": source,
        "micro_sam": importlib.metadata.version("micro_sam"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[30.0, 30.0, 30.0], metavar=("Z", "Y", "X"),
                        help="The voxel size to segment at, in nm.")
    parser.add_argument("--modes", nargs="+", choices=["xy"], default=["xy"],
                        help="Only 'xy'; accepted so the arguments of a sweep's best.json can be passed on.")
    parser.add_argument("--invert", action="store_true", help="Invert the contrast, i.e. segment 255 - raw.")
    parser.add_argument("--model_type", default=MODEL_TYPE, help="The micro-sam model, with an AIS decoder.")
    parser.add_argument("--test_split", default=TEST_SPLIT, help="The folder with the held-out test blocks.")
    parser.add_argument("--split_file", help="Segment the blocks of a split file instead of the test split.")
    parser.add_argument("--split_key", default="val", help="The key of the split file to segment.")
    parser.add_argument("--segmentation_root", default=SEGMENTATION_ROOT,
                        help="Where the segmentations are written, one folder per setting.")
    parser.add_argument("--force", action="store_true", help="Recompute segmentations that already exist.")
    args = parser.parse_args()

    import torch
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    if args.split_file:
        blocks = find_split_blocks(args.split_file, args.split_key)
        source = f"{args.split_file}:{args.split_key}"
    else:
        blocks = find_test_blocks(args.test_split)
        source = args.test_split
    voxel_size = tuple(args.voxel_size)
    min_size = min_size_for(voxel_size)
    prefix = setting_prefix("microsam", voxel_size, args.invert)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{prefix}: micro-sam {args.model_type} at {voxel_size} nm on {len(blocks)} block(s), "
          f"device {device}, min_size {min_size}\n")

    segmenters = {}
    for dataset, block_path in blocks.items():
        output_path = os.path.join(args.segmentation_root, f"{prefix}-xy", f"{dataset}.h5")
        if not args.force and os.path.exists(output_path):
            print(f"  {dataset}: already segmented, skipping")
            continue

        with h5py.File(block_path, "r") as f:
            raw = f["raw"][:]
        if args.invert:
            raw = np.iinfo(raw.dtype).max - raw
        volume = to_target_resolution(raw, voxel_size)

        tiled = max(volume.shape[1:]) > SAM_INPUT
        if tiled not in segmenters:
            segmenters[tiled] = get_predictor_and_segmenter(
                args.model_type, segmentation_mode="ais", is_tiled=tiled,
            )
        predictor, segmenter = segmenters[tiled]
        seg, canvas_shape, tile_shape, halo = segment_microsam(predictor, segmenter, volume, min_size)
        print(f"  {dataset}: {raw.shape} at {NATIVE_VOXEL_SIZE} nm -> {volume.shape}, "
              f"SAM canvas {canvas_shape}" + (f", tiles {tile_shape} + halo {halo}" if tile_shape else ""),
              flush=True)

        native = to_native_resolution(seg, raw.shape)
        n_lost = n_objects(seg) - n_objects(native)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as f:
            ds = f.create_dataset("seg", data=native, compression="gzip")
            ds.attrs.update(settings(args.model_type, voxel_size, args.invert, min_size, seg.shape, raw.shape,
                                     canvas_shape, tile_shape, halo, n_lost, source))
            f.create_dataset("seg_inference", data=seg, compression="gzip")
        lost = f", {n_lost} lost on the native grid" if n_lost else ""
        print(f"    xy: {n_objects(seg)} objects{lost} -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
