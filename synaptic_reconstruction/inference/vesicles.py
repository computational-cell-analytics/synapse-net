import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np

import torch

from synaptic_reconstruction.inference.util import apply_size_filter, get_prediction, _Scaler
from synaptic_reconstruction.inference.postprocessing.vesicles import filter_border_objects


def distance_based_vesicle_segmentation(
    foreground: np.ndarray,
    boundaries: np.ndarray,
    verbose: bool,
    min_size: int,
    boundary_threshold: float = 0.5,  # previous default value was 0.9
    distance_threshold: int = 8,
    block_shape: Tuple[int, int, int] = (128, 256, 256),
    halo: Tuple[int, int, int] = (48, 48, 48),
) -> np.ndarray:
    """Segment vesicles using a seeded watershed from connected components derived from
    distance transform of the boundary predictions.

    This approach can prevent false merges that occur with the `simple_vesicle_segmentation`.

    Args:
        foreground: The foreground prediction.
        boundaries: The boundary prediction.
        verbose: Whether to print timing information.
        min_size: The minimal vesicle size.
        boundary_threshold: The threshold for binarizing the boundary predictions for the distance computation.
        distance_threshold: The threshold for finding connected components in the boundary distances.
        block_shape: Block shape for parallelizing the operations.
        halo: Halo for parallelizing the operations.

    Returns:
        The vesicle segmentation.
    """
    # Compute the boundary distances.
    t0 = time.time()
    bd_dist = parallel.distance_transform(
        boundaries < boundary_threshold, halo=halo, verbose=verbose, block_shape=block_shape
    )
    bd_dist[foreground < 0.5] = 0
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    # Get the segmentation via seeded watershed of components in the boundary distances.
    t0 = time.time()
    print(f"using a distance thresholf of {distance_threshold} for distance based segmentation")
    seeds = parallel.label(bd_dist > distance_threshold, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # Compute distances from the seeds, which are used as heightmap for the watershed,
    # to assign all pixels to the nearest seed.
    t0 = time.time()
    dist = parallel.distance_transform(seeds == 0, halo=halo, verbose=verbose, block_shape=block_shape)
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    t0 = time.time()
    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        dist, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    seg = apply_size_filter(seg, min_size, verbose, block_shape)
    return seg


def simple_vesicle_segmentation(
    foreground: np.ndarray,
    boundaries: np.ndarray,
    verbose: bool,
    min_size: int,
    block_shape: Tuple[int, int, int] = (128, 256, 256),
    halo: Tuple[int, int, int] = (48, 48, 48),
) -> np.ndarray:
    """Segment vesicles by subtracting boundary from foreground prediction and
    applying connected components.

    Args:
        foreground: The foreground prediction.
        boundaries: The boundary prediction.
        verbose: Whether to print timing information.
        min_size: The minimal vesicle size.
        block_shape: Block shape for parallelizing the operations.
        halo: Halo for parallelizing the operations.

    Returns:
        The vesicle segmentation.
    """

    t0 = time.time()
    seeds = parallel.label((foreground - boundaries) > 0.5, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    t0 = time.time()
    dist = parallel.distance_transform(seeds == 0, halo=halo, verbose=verbose, block_shape=block_shape)
    if verbose:
        print("Compute distance transform in", time.time() - t0, "s")

    t0 = time.time()
    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        dist, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=verbose, halo=halo,
    )
    if verbose:
        print("Compute watershed in", time.time() - t0, "s")

    seg = apply_size_filter(seg, min_size, verbose, block_shape)
    return seg


def segment_vesicles(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    min_size: int = 500,
    verbose: bool = True,
    distance_based_segmentation: bool = True,
    distance_threshold: int = 8,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    exclude_boundary: bool = False,
    mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Segment vesicles in an input volume or image.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a vesicle to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        exclude_boundary: Whether to exclude vesicles that touch the upper / lower border in z.
        mask: An optional mask that is used to restrict the segmentation.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting vesicles in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Rescale the mask if it was given and run prediction.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(input_volume, tiling=tiling, model_path=model_path, model=model, verbose=verbose, mask=mask)
    foreground, boundaries = pred[:2]

    # Deal with 2D segmentation case.
    kwargs = {}
    if len(input_volume.shape) == 2:
        kwargs["block_shape"] = (256, 256)
        kwargs["halo"] = (48, 48)

    if distance_based_segmentation:
        seg = distance_based_vesicle_segmentation(
            foreground, boundaries, verbose=verbose, min_size=min_size, distance_threshold = distance_threshold, **kwargs
        )
    else:
        seg = simple_vesicle_segmentation(
            foreground, boundaries, verbose=verbose, min_size=min_size, **kwargs
        )

    if exclude_boundary:
        seg = filter_border_objects(seg)
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
