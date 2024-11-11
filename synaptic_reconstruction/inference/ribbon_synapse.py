from typing import Dict, Sequence, Optional, Union

import numpy as np
import torch

from synaptic_reconstruction.inference.util import get_prediction, _Scaler


def segment_ribbon_synapse_structures(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    structure_names: Sequence[str] = ("ribbon", "PD", "membrane"),
    verbose: bool = False,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    threshold: Optional[Union[float, Dict[str, float]]] = None,
    scale: Optional[Sequence[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Segment ribbon synapse structures.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        structure_names: Names of the structures to be segmented.
            The default network segments the ribbon, presynaptic density (pd) an local memrane.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        threshold: The threshold for binarizing predictions.
        scale: The scale factor to use for rescaling the input volume before prediction.
        mask: An optional mask that is used to restrict the segmentation.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting ribbon synapse structures in volume of shape", input_volume.shape)
    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    if mask is not None:
        mask = scaler.scale_input(mask, is_segmentation=True)
    predictions = get_prediction(
        input_volume, model_path=model_path, model=model, tiling=tiling, mask=mask, verbose=verbose
    )
    assert len(structure_names) == predictions.shape[0]

    predictions = {
        name: scaler.rescale_output(predictions[i], is_segmentation=False) for i, name in enumerate(structure_names)
    }
    if threshold is not None:
        for name in structure_names:
            # We can either have a single threshold value or a threshold per structure
            # that is given as a dictionary.
            this_threshold = threshold if isinstance(threshold, float) else threshold[name]
            predictions[name] = predictions[name] > this_threshold

    return predictions
