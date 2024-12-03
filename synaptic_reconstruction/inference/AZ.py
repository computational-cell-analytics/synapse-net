import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np
import torch

from synaptic_reconstruction.inference.util import get_prediction, _Scaler
from synaptic_reconstruction.inference.postprocessing.postprocess_AZ import find_intersection_boundary

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

def segment_AZ(
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
    """
    Segment mitochondria in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.

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
    #print(f"shape {foreground.shape}")
    #foreground = pred[0, :, :, :]
    print(f"shape {foreground.shape}")

    segmentation = _run_segmentation(foreground, verbose=verbose, min_size=min_size)

    #returning prediciton and intersection not possible atm, but currently do not need prediction anyways
    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return segmentation, pred

    if compartment is not None:
        intersection = find_intersection_boundary(segmentation, compartment)
        return segmentation, intersection
        
    return segmentation

