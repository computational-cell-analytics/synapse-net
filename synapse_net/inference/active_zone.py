import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np
import torch

from skimage.segmentation import find_boundaries
from synapse_net.inference.util import get_prediction, _Scaler


def find_intersection_boundary(segmented_AZ: np.ndarray, segmented_compartment: np.ndarray) -> np.ndarray:
    """Find the intersection of the boundaries of each objects in segmented_compartment with segmented_AZ.

    Args:
        segmented_AZ: 3D array representing the active zone (AZ).
        segmented_compartment: 3D array representing the compartment, with multiple labels.

    Returns:
        Array with the cumulative intersection of all boundaries of segmented_compartment labels with segmented_AZ.
    """
    # Step 0: Initialize an empty array to accumulate intersections
    cumulative_intersection = np.zeros_like(segmented_AZ, dtype=bool)

    # Step 1: Loop through each unique label in segmented_compartment (excluding 0 if it represents background)
    labels = np.unique(segmented_compartment)
    labels = labels[labels != 0]  # Exclude background label (0) if necessary

    for label in labels:
        # Step 2: Create a binary mask for the current label
        label_mask = (segmented_compartment == label)

        # Step 3: Find the boundary of the current label's compartment
        boundary_compartment = find_boundaries(label_mask, mode='outer')

        # Step 4: Find the intersection with the AZ for this label's boundary
        intersection = np.logical_and(boundary_compartment, segmented_AZ)

        # Step 5: Accumulate intersections for each label
        cumulative_intersection = np.logical_or(cumulative_intersection, intersection)

    return cumulative_intersection.astype(int)  # Convert boolean array to int (1 for intersecting points, 0 elsewhere)


def _run_segmentation(
    foreground, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
):

    # get the segmentation via seeded watershed
    t0 = time.time()
    seg = parallel.label(foreground > 0.5, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # size filter
    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    seg = np.where(seg > 0, 1, 0)
    return seg


def segment_active_zone(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    min_size: int = 500,
    verbose: bool = True,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
    compartment: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Segment active zones in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.
        compartment: Pass a compartment segmentation, to intersect the boundaries of the
            compartments with the active zone prediction.

    Returns:
        The foreground mask as a numpy array.
    """
    if verbose:
        print("Segmenting AZ in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Rescale the mask if it was given and run prediction.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(input_volume, model_path=model_path, model=model, tiling=tiling, mask=mask, verbose=verbose)

    # Run segmentation and rescale the result if necessary.
    foreground = pred[0]
    print(f"shape {foreground.shape}")

    segmentation = _run_segmentation(foreground, verbose=verbose, min_size=min_size)
    segmentation = scaler.rescale_output(segmentation, is_segmentation=True)

    # Returning prediciton and intersection currently not possible.
    if return_predictions:
        assert compartment is None
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return segmentation, pred

    if compartment is not None:
        assert not return_predictions
        compartment = scaler.scale_input(input_volume, is_segmentation=True)
        intersection = find_intersection_boundary(segmentation, compartment)
        return segmentation, intersection

    return segmentation
