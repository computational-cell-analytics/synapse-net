import os
import tempfile
from typing import Dict, List, Optional

import elf.parallel as parallel
import numpy as np
import torch

from elf.io import open_file
from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper
from elf.wrapper.base import MultiTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
from numpy.typing import ArrayLike
from synapse_net.inference.util import get_prediction


class SelectChannel(SimpleTransformationWrapper):
    """Wrapper to select a chanel from an array-like dataset object.

    Args:
        volume: The array-like input dataset.
        channel: The channel that will be selected.
    """
    def __init__(self, volume: np.typing.ArrayLike, channel: int):
        self.channel = channel
        super().__init__(volume, lambda x: x[self.channel], with_channels=True)

    @property
    def shape(self):
        return self._volume.shape[1:]

    @property
    def chunks(self):
        return self._volume.chunks[1:]

    @property
    def ndim(self):
        return self._volume.ndim - 1


def _run_segmentation(pred, output, seeds, chunks, seed_threshold, min_size, verbose, original_shape):
    # Create wrappers for selecting the foreground and the boundary channel.
    foreground = SelectChannel(pred, 0)
    boundaries = SelectChannel(pred, 1)

    # Create wrappers for subtracting and thresholding boundary subtracted from the foreground.
    # And then compute the seeds based on this.
    seed_input = ThresholdWrapper(
        MultiTransformationWrapper(np.subtract, foreground, boundaries), seed_threshold
    )
    parallel.label(seed_input, seeds, verbose=verbose, block_shape=chunks)

    # Run watershed to extend back from the seeds to the boundaries.
    mask = ThresholdWrapper(foreground, 0.5)

    # Resize if necessary.
    if original_shape is not None:
        boundaries = ResizedVolume(boundaries, original_shape, order=1)
        seeds = ResizedVolume(seeds, original_shape, order=0)
        mask = ResizedVolume(mask, original_shape, order=0)

    parallel.seeded_watershed(
        boundaries, seeds=seeds, out=output, verbose=verbose, mask=mask, block_shape=chunks, halo=3 * (16,)
    )

    # Run the size filter.
    if min_size > 0:
        parallel.size_filter(output, output, min_size=min_size, verbose=verbose, block_shape=chunks)


def scalable_segmentation(
    input_: ArrayLike,
    output: ArrayLike,
    model: torch.nn.Module,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    scale: Optional[List[float]] = None,
    seed_threshold: float = 0.5,
    min_size: int = 500,
    prediction: Optional[ArrayLike] = None,
    verbose: bool = True,
    mask: Optional[ArrayLike] = None,
) -> None:
    """Run segmentation based on a prediction with foreground and boundary channel.

    This function first subtracts the boundary prediction from the foreground prediction,
    then applies a threshold, connected components, and a watershed to fit the components
    back to the foreground. All processing steps are implemented in a scalable fashion,
    so that the function runs for large input volumes.

    Args:
        input_: The input data.
        output: The array for storing the output segmentation.
            Can be a numpy array, a zarr array, or similar.
        model: The model for prediction.
        tiling: The tiling configuration for the prediction.
        scale: The scale factor to use for rescaling the input volume before prediction.
        seed_threshold: The threshold applied before computing connected components.
        min_size: The minimum size of a vesicle to be considered.
        prediction: The array for storing the prediction.
            If given, this can be a numpy array, a zarr array, or similar
            If not given will be stored in a temporary n5 array.
        verbose: Whether to print timing information.
    """
    if mask is not None:
        raise NotImplementedError
    assert model.out_channels == 2

    # Create a temporary directory for storing the predictions.
    chunks = (128,) * 3
    with tempfile.TemporaryDirectory() as tmp_dir:

        if scale is None or np.allclose(scale, 1.0, atol=1e-3):
            original_shape = None
        else:
            original_shape = input_.shape
            new_shape = tuple(int(sh * sc) for sh, sc in zip(input_.shape, scale))
            input_ = ResizedVolume(input_, shape=new_shape, order=1)

        if prediction is None:
            # Create the dataset for storing the prediction.
            tmp_pred = os.path.join(tmp_dir, "prediction.n5")
            f = open_file(tmp_pred, mode="a")
            pred_shape = (2,) + input_.shape
            pred_chunks = (1,) + chunks
            prediction = f.create_dataset("pred", shape=pred_shape, dtype="float32", chunks=pred_chunks)
        else:
            assert prediction.shape[0] == 2
            assert prediction.shape[1:] == input_.shape

        # Create temporary storage for the seeds.
        tmp_seeds = os.path.join(tmp_dir, "seeds.n5")
        f = open_file(tmp_seeds, mode="a")
        seeds = f.create_dataset("seeds", shape=input_.shape, dtype="uint64", chunks=chunks)

        # Run prediction and segmentation.
        get_prediction(input_, prediction=prediction, tiling=tiling, model=model, verbose=verbose)
        _run_segmentation(prediction, output, seeds, chunks, seed_threshold, min_size, verbose, original_shape)
