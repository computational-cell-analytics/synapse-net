from typing import Optional, Dict, Union, Tuple

import numpy as np

from synaptic_reconstruction.inference.util import get_prediction, get_default_tiling
from skimage.measure import label


# TODO: How exactly do we post-process the actin?
# Do we want to run an instance segmentation to extract
# individual fibers?
# For now we only do connected components to remove small
# fragments and then binarize again.
def segment_actin(
    input_volume: np.ndarray,
    model_path: str,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    foreground_threshold: float = 0.5,
    min_size: int = 0,
    verbose: bool = True,
    return_predictions: bool = False,
    exclude_boundary: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    if tiling is None:
        tiling = get_default_tiling()

    pred = get_prediction(input_volume, model_path, tiling, verbose)
    foreground, boundaries = pred[:2]

    # TODO proper segmentation procedure
    # NOTE: actin fiber recall may improve by choosing a lower foreground threshold
    seg = foreground > foreground_threshold
    if min_size > 0:
        seg = label(seg)
        ids, sizes = np.unique(seg, return_counts=True)
        filter_ids = ids[sizes < min_size]
        seg[np.isin(seg, filter_ids)] = 0
        seg = (seg > 0).astype("uint8")

    if return_predictions:
        return seg, foreground
    return seg
