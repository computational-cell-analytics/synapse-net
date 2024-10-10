from typing import Optional, Dict, List, Union, Tuple

import numpy as np
import torch

from skimage.measure import label
from synaptic_reconstruction.inference.util import apply_size_filter, get_prediction, _Scaler


# TODO: How exactly do we post-process the actin?
# Do we want to run an instance segmentation to extract
# individual fibers?
# For now we only do connected components to remove small
# fragments and then binarize again.
def segment_actin(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    foreground_threshold: float = 0.5,
    min_size: int = 0,
    verbose: bool = True,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment actin in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        foreground_threshold: Threshold for binarizing foreground predictions.
        min_size: The minimum size of an actin fiber to be considered.
        verbose: Whether to print timing information.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting actin in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Run the prediction.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(input_volume, model=model, model_path=model_path, tiling=tiling, verbose=verbose)
    foreground, boundaries = pred[:2]

    # TODO proper segmentation procedure
    # NOTE: actin fiber recall may improve by choosing a lower foreground threshold
    seg = foreground > foreground_threshold
    if min_size > 0:
        seg = label(seg)
        seg = apply_size_filter(seg, min_size, verbose=verbose)
        seg = (seg > 0).astype("uint8")
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        foreground = scaler.rescale_output(foreground, is_segmentation=True)
        return seg, foreground
    return seg
