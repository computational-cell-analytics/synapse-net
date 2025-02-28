import time
from typing import Dict, List, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np
import torch

from synapse_net.inference.util import get_prediction, _Scaler


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
    with_channels = kwargs.pop("with_channels", True)
    channels_to_standardize = kwargs.pop("channels_to_standardize", [0])
    if verbose:
        print("Segmenting cristae in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Run prediction and segmentation.
    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    pred = get_prediction(
        input_volume, model_path=model_path, model=model, mask=mask,
        tiling=tiling, with_channels=with_channels, channels_to_standardize=channels_to_standardize, verbose=verbose
    )
    foreground, boundaries = pred[:2]
    seg = _run_segmentation(foreground, verbose=verbose, min_size=min_size)
    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
