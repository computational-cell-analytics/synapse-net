"""Segment volume EM blocks with MitoNet at a chosen voxel size.

This is the MitoNet baseline for compare_models.py. MitoNet (empanada) is a 2D generalist, and its 3D
mode, the ortho-plane consensus, infers on the xy, xz and yz planes, so it assumes isotropic voxels.
The blocks are at 25 nm in z and 5 nm in xy. Each block is resampled to the target voxel size,
segmented there, and the result is resized back to the native grid with nearest-neighbor
interpolation. The ground truth is never resampled; compare_models.py scores the native 'seg'
against it with '-s', and score_mitonet_sweep.py scores a whole voxel-size sweep.

Two modes can come out of the same run, and each is written to its own folder:
  <prefix>-ortho  the ortho-plane consensus of the xy, xz and yz stacks; only meaningful isotropic
  <prefix>-xy     the xy stack alone, the mode of the earlier MitoNet runs
The prefix names the setting: 'mitonet-iso30' for 30 nm isotropic, 'mitonet-z25-xy10' for 25 nm in z
and 10 nm in xy, with '-inv' appended when the contrast is inverted.

The size filter is matched to synapse-net in physical volume rather than left at empanada's default,
see voxel_sweep.min_size_for. All other parameters are the Engine3d defaults.

This runs in the empanada environment, which has neither synapse_net nor torch_em nor elf, so it does
not import compare_models.py or synapse_net; the block lookup and resampling come from voxel_sweep.py.

Usage:
    python segment_mitonet.py [--voxel_size Z Y X] [--modes ortho xy] [--invert] [--force]
    python segment_mitonet.py --split_file SPLIT.json --split_key val --segmentation_root ROOT ...
"""

import argparse
import importlib.metadata
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel_sweep import (  # noqa: E402
    NATIVE_VOXEL_SIZE, TEST_SPLIT, find_split_blocks, find_test_blocks, min_size_for, n_objects,
    setting_prefix, to_native_resolution, to_target_resolution,
)

SEGMENTATION_ROOT = "/mnt/lustre-grete/usr/u15205/volume-em/repro-comparison"
MIN_EXTENT = 4

# The Engine3d defaults, which the earlier MitoNet runs used too. They are set explicitly so the
# settings recorded with the segmentation stay true if a newer empanada changes its defaults. The
# napari widget has different defaults (confidence_thr 0.5, median_kernel_size 3, min_extent 5).
ENGINE_PARAMS = dict(
    inference_scale=1, confidence_thr=0.3, median_kernel_size=5,
    nms_threshold=0.1, nms_kernel=3, label_divisor=1000,
)
# The consensus defaults, which the napari widget uses as well.
CONSENSUS_PARAMS = dict(pixel_vote_thr=2, cluster_iou_thr=0.75, allow_one_view=False)

# MitoNet_v1.yaml, without the FINETUNE block that inference does not read.
MITONET_CONFIG = {
    "class_names": {1: "mito"},
    "labels": [1],
    "thing_list": [1],
    "model": "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1",
    "model_quantized": "https://zenodo.org/record/6861565/files/MitoNet_v1_quantized.pth?download=1",
    "padding_factor": 16,
    "norms": {"mean": 0.57571, "std": 0.12765},
}

MODES = ("ortho", "xy")


def segment_mitonet(engine, volume, modes, min_size):
    """Segment a volume with MitoNet and return a segmentation per mode."""
    from empanada_napari.inference import stack_postprocessing, tracker_consensus

    axes = ("xy", "xz", "yz") if "ortho" in modes else ("xy",)
    trackers = {}
    for axis_name in axes:
        _, trackers[axis_name] = engine.infer_on_axis(volume, axis_name)

    # Both functions are napari thread workers; call the generators they wrap, which yield one
    # (volume, class name, instances) per class. MitoNet has only the one class.
    segmentations = {}
    if "xy" in modes:
        segmentations["xy"], _, _ = next(stack_postprocessing.__wrapped__(
            {"xy": trackers["xy"]}, None, MITONET_CONFIG, label_divisor=engine.label_divisor,
            min_size=min_size, min_extent=MIN_EXTENT, dtype=np.uint32,
        ))
    if "ortho" in modes:
        segmentations["ortho"], _, _ = next(tracker_consensus.__wrapped__(
            trackers, None, MITONET_CONFIG, label_divisor=engine.label_divisor, **CONSENSUS_PARAMS,
            min_size=min_size, min_extent=MIN_EXTENT, dtype=np.uint32,
        ))
    return segmentations


def settings(mode, voxel_size, invert, min_size, inference_shape, native_shape, n_lost, source):
    """The settings a segmentation was made with, stored as the attributes of its 'seg'."""
    effective = [n * size / m for n, size, m in zip(native_shape, NATIVE_VOXEL_SIZE, inference_shape)]
    attrs = {
        "model": "MitoNet_v1",
        "mode": "ortho-plane consensus of xy, xz, yz" if mode == "ortho" else "xy stack",
        "target_voxel_size_nm": np.array(voxel_size, dtype=float),
        "inference_voxel_size_nm": np.round(effective, 2),
        "invert": bool(invert),
        "min_size": min_size,
        "min_extent": MIN_EXTENT,
        **{key: value for key, value in ENGINE_PARAMS.items() if key != "inference_scale"},
    }
    if mode == "ortho":
        attrs.update(CONSENSUS_PARAMS)
    attrs["n_lost_to_native"] = n_lost
    attrs["blocks"] = source
    attrs["empanada_napari"] = importlib.metadata.version("empanada-napari")
    return attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[30.0, 30.0, 30.0], metavar=("Z", "Y", "X"),
                        help="The voxel size to segment at, in nm.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                        help="The modes to write. 'xy' alone skips the xz and yz inference.")
    parser.add_argument("--invert", action="store_true", help="Invert the contrast, i.e. segment 255 - raw.")
    parser.add_argument("--test_split", default=TEST_SPLIT, help="The folder with the held-out test blocks.")
    parser.add_argument("--split_file", help="Segment the blocks of a split file instead of the test split.")
    parser.add_argument("--split_key", default="val", help="The key of the split file to segment.")
    parser.add_argument("--segmentation_root", default=SEGMENTATION_ROOT,
                        help="Where the segmentations are written, one folder per setting and mode.")
    parser.add_argument("--force", action="store_true", help="Recompute segmentations that already exist.")
    args = parser.parse_args()

    from empanada_napari.inference import Engine3d

    if args.split_file:
        blocks = find_split_blocks(args.split_file, args.split_key)
        source = f"{args.split_file}:{args.split_key}"
    else:
        blocks = find_test_blocks(args.test_split)
        source = args.test_split
    voxel_size = tuple(args.voxel_size)
    modes = [mode for mode in MODES if mode in args.modes]
    if "ortho" in modes and len(set(voxel_size)) > 1:
        print(f"Warning: ortho-plane inference at the anisotropic voxel size {voxel_size}.")
    min_size = min_size_for(voxel_size)
    prefix = setting_prefix("mitonet", voxel_size, args.invert)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{prefix}: MitoNet at {voxel_size} nm on {len(blocks)} block(s), modes {', '.join(modes)}, "
          f"device {device}, min_size {min_size}, min_extent {MIN_EXTENT}\n")

    engine = None
    for dataset, block_path in blocks.items():
        output_paths = {mode: os.path.join(args.segmentation_root, f"{prefix}-{mode}", f"{dataset}.h5")
                        for mode in modes}
        if not args.force and all(os.path.exists(path) for path in output_paths.values()):
            print(f"  {dataset}: already segmented, skipping")
            continue

        with h5py.File(block_path, "r") as f:
            raw = f["raw"][:]
        if args.invert:
            raw = np.iinfo(raw.dtype).max - raw
        volume = to_target_resolution(raw, voxel_size)
        print(f"  {dataset}: {raw.shape} at {NATIVE_VOXEL_SIZE} nm -> {volume.shape}", flush=True)

        if engine is None:
            engine = Engine3d(MITONET_CONFIG, min_size=min_size, min_extent=MIN_EXTENT, **ENGINE_PARAMS)
        segmentations = segment_mitonet(engine, volume, modes, min_size)

        for mode, seg in segmentations.items():
            native = to_native_resolution(seg, raw.shape)
            n_lost = n_objects(seg) - n_objects(native)
            output_path = output_paths[mode]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f:
                ds = f.create_dataset("seg", data=native, compression="gzip")
                ds.attrs.update(settings(mode, voxel_size, args.invert, min_size, seg.shape, raw.shape,
                                         n_lost, source))
                f.create_dataset("seg_inference", data=seg.astype(np.uint32), compression="gzip")
            lost = f", {n_lost} lost on the native grid" if n_lost else ""
            print(f"    {mode}: {n_objects(seg)} objects{lost} -> {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
