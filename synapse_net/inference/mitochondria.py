import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np
import torch

from synapse_net.inference.util import apply_size_filter, get_prediction, _Scaler, _postprocess_seg_3d


def _run_segmentation(
    foreground, boundaries, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    area_threshold=5000,
):
    t0 = time.time()
    dist = parallel.distance_transform(
        boundaries < boundary_threshold, halo=halo, verbose=verbose, block_shape=block_shape
    )
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    # Get the segmentation via seeded watershed.
    t0 = time.time()
    seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    t0 = time.time()
    hmap = (dist.max() - dist) / dist.max()
    hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()
    mask = (foreground + boundaries) > 0.5

    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    seg = apply_size_filter(seg, min_size, verbose=verbose, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold)
    return seg


def segment_mitochondria(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    min_size: int = 50000,
    verbose: bool = True,
    distance_based_segmentation: bool = False,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
    seed_distance: int = 6,
    ws_block_shape: Tuple[int, ...] = (128, 256, 256),
    ws_halo: Tuple[int, ...] = (48, 48, 48),
    boundary_threshold: float = 0.25,
    area_threshold: int = 5000,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Segment mitochondria in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a mitochondria to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.
        seed_distance: The distance threshold for the seeded watershed.
        ws_block_shape: The block shape for the seeded watershed.
        ws_halo: The halo for the seeded watershed.
        boundary_threshold: The boundary threshold distance calculation.
        area_threshold: The maximum area (in pixels) of holes to be removed or filled in the segmentation.
            This parameter is passed to `skimage.morphology.remove_small_holes`.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting mitochondria in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Rescale the mask if it was given and run prediction.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(input_volume, model_path=model_path, model=model, tiling=tiling, mask=mask, verbose=verbose)

    # Run segmentation and rescale the result if necessary.
    foreground, boundaries = pred[:2]
    seg = _run_segmentation(foreground, boundaries, verbose=verbose, min_size=min_size, seed_distance=seed_distance,
                            block_shape=ws_block_shape, halo=ws_halo, boundary_threshold=boundary_threshold,
                            area_threshold=area_threshold)
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
