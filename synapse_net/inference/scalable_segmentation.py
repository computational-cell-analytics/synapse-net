import os
import tempfile
from typing import Dict, Optional

import elf.parallel as parallel
import numpy as np
import torch

from elf.io import open_file
from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper
from elf.wrapper.base import MultiTransformationWrapper
from synapse_net.inference.util import get_prediction
from numpy.typing import ArrayLike


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


def _run_segmentation(pred, output, seeds, chunks, seed_threshold, min_size, verbose):
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
    parallel.seeded_watershed(
        boundaries, seeds=seeds, out=output, verbose=verbose, mask=mask, block_shape=chunks, halo=3 * (16,)
    )

    # Run the size filter.
    if min_size > 0:
        parallel.size_filter(output, output, min_size=min_size, verbose=verbose, block_shape=chunks)


# TODO support resizing via the wrapper
def scalable_segmentation(
    input_: ArrayLike,
    output: ArrayLike,
    model: torch.nn.Module,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    seed_threshold: float = 0.5,
    min_size: int = 500,
    verbose: bool = True,
) -> None:
    """Lorem ipsum

    Args:
        input_:
        output:
        model: The model.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a vesicle to be considered.
        verbose: Whether to print timing information.
    """
    assert model.out_channels == 2

    # Create a temporary directory for storing the predictions.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pred = os.path.join(tmp_dir, "prediction.n5")
        f = open_file(tmp_pred, mode="a")

        # Create the dataset for storing the prediction.
        chunks = (128,) * 3
        pred_shape = (2,) + input_.shape
        pred_chunks = (1,) + chunks
        pred = f.create_dataset("pred", shape=pred_shape, dtype="float32", chunks=pred_chunks)

        # Create temporary storage for the seeds.
        tmp_seeds = os.path.join(tmp_dir, "seeds.n5")
        f = open_file(tmp_seeds, mode="a")
        seeds = f.create_dataset("seeds", shape=input_.shape, dtype="uint64", chunks=chunks)

        # Run prediction and segmentation.
        get_prediction(input_, prediction=pred, tiling=tiling, model=model, verbose=verbose)
        _run_segmentation(pred, output, seeds, chunks, seed_threshold, min_size, verbose)
