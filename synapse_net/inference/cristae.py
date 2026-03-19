import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
from elf.wrapper.base import (
    SimpleTransformationWrapper,
    MultiTransformationWrapper,
)
from skimage.morphology import binary_erosion, ball
from skimage.measure import regionprops
import numpy as np
import torch

from synapse_net.inference.util import get_prediction, _Scaler


def _erode_instances(mito_data, erode_voxels, verbose):
    """Erodes instances globally and returns a memory-efficient boolean mask."""
    if verbose:
        t_erode = time.time()
        print("Eroding mitochondria instances globally...")

    footprint = ball(erode_voxels)
    props = regionprops(mito_data)

    # Allocate a boolean array
    eroded_binary_mask = np.zeros(mito_data.shape, dtype=bool)

    for prop in props:
        sl = prop.slice

        # Isolate this specific instance within its bounding box
        instance_mask = (mito_data[sl] == prop.label)

        # Apply erosion
        eroded_mask = binary_erosion(instance_mask, footprint=footprint)

        # Write True to the newly eroded locations in our boolean array
        eroded_binary_mask[sl][eroded_mask] = True

    if verbose:
        print(f"Instance erosion completed in {time.time() - t_erode:.2f} s")

    return eroded_binary_mask


def _run_segmentation(
    foreground, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
    mito_seg=None,
    erode_voxels=3,
):
    mito_seg = _erode_instances(mito_seg, erode_voxels, verbose)

    # Mask the foreground lazily
    # Even though mito_seg is now in memory, foreground might not be.
    # MultiTransformationWrapper safely handles this mix.
    def mask_foreground(inputs):
        fg_block, mito_block = inputs
        return np.where(mito_block != 0, fg_block, 0)

    foreground = MultiTransformationWrapper(
        mask_foreground,
        foreground,
        mito_seg,
        apply_to_list=True
    )

    # Apply the threshold lazily
    def threshold_block(block):
        return block > 0.5

    binary_foreground = SimpleTransformationWrapper(
        foreground,
        transformation=threshold_block
    )

    t0 = time.time()
    seg = parallel.label(binary_foreground, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # Size filter
    if min_size > 0:
        t0 = time.time()
        parallel.size_filter(seg, out=seg, min_size=min_size, block_shape=block_shape, verbose=verbose)
        if verbose:
            print("Size filter in", time.time() - t0, "s")

    seg = np.where(seg > 0, 1, 0)

    return seg


def segment_cristae(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    min_size: int = 500,
    verbose: bool = True,
    distance_based_segmentation: bool = False,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Segment cristae in an input volume.

    Args:
        input_volume: The input volume to segment. Expects 2 3D volumes: raw and mitochondria
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a cristae to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    verbose = True
    mitochondria = kwargs.pop("extra_segmentation", None)
    if mitochondria is None:
        # try extract from input volume
        if input_volume.ndim == 4:
            mitochondria = input_volume[1]
            input_volume = input_volume[0]
    if mitochondria is None:
        raise ValueError("Mitochondria segmentation is required")
    with_channels = kwargs.pop("with_channels", True)
    channels_to_standardize = kwargs.pop("channels_to_standardize", [0])
    if verbose:
        print("Segmenting cristae in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    # rescale each channel
    volume = scaler.scale_input(input_volume)
    mito_seg = scaler.scale_input(mitochondria, is_segmentation=True)
    input_volume = np.stack([volume, mito_seg], axis=0)

    # Run prediction and segmentation.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(
        input_volume, model_path=model_path, model=model, mask=mask,
        tiling=tiling, with_channels=with_channels, channels_to_standardize=channels_to_standardize, verbose=verbose
    )
    foreground, boundaries = pred[:2]
    seg = _run_segmentation(foreground, verbose=verbose, min_size=min_size, mito_seg=mito_seg)
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
