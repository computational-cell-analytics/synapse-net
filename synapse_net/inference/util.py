import os
import time
import warnings
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

# # Suppress annoying import warnings.
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import bioimageio.core

import imageio.v3 as imageio
import elf.parallel as parallel
import mrcfile
import numpy as np
import torch
import torch_em
# import xarray

from elf.io import open_file
from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes
from skimage.transform import rescale, resize
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm


#
# Utils for prediction.
#


class _Scaler:
    def __init__(self, scale, verbose):
        self.verbose = verbose
        self._original_shape = None

        if scale is None:
            self.scale = None
            return

        # Convert scale to a NumPy array (ensures consistency)
        scale = np.atleast_1d(scale).astype(np.float64)

        # Validate scale values
        if not np.issubdtype(scale.dtype, np.number):
            raise TypeError(f"Scale contains non-numeric values: {scale}")

        # Check if scaling is effectively identity (1.0 in all dimensions)
        if np.allclose(scale, 1.0, atol=1e-3):
            self.scale = None
        else:
            self.scale = scale

    def scale_input(self, input_volume, is_segmentation=False):
        t0 = time.time()
        if self.scale is None:
            return input_volume

        if self._original_shape is None:
            self._original_shape = input_volume.shape
        elif self._original_shape != input_volume.shape:
            raise RuntimeError(
                "Scaler was called with different input shapes. "
                "This is not supported, please create a new instance of the class for it."
            )

        if is_segmentation:
            input_volume = rescale(
                input_volume, self.scale, preserve_range=True, order=0, anti_aliasing=False,
            ).astype(input_volume.dtype)
        else:
            input_volume = rescale(input_volume, self.scale, preserve_range=True).astype(input_volume.dtype)

        if self.verbose:
            print("Rescaled volume from", self._original_shape, "to", input_volume.shape, "in", time.time() - t0, "s")
        return input_volume

    def rescale_output(self, output, is_segmentation):
        t0 = time.time()
        if self.scale is None:
            return output

        assert self._original_shape is not None
        out_shape = self._original_shape
        if output.ndim > len(out_shape):
            assert output.ndim == len(out_shape) + 1
            out_shape = (output.shape[0],) + out_shape

        if is_segmentation:
            output = resize(output, out_shape, preserve_range=True, order=0, anti_aliasing=False).astype(output.dtype)
        else:
            output = resize(output, out_shape, preserve_range=True).astype(output.dtype)

        if self.verbose:
            print("Resized prediction back to original shape", output.shape, "in", time.time() - t0, "s")

        return output


def get_prediction(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Optional[Dict[str, Dict[str, int]]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
    channels_to_standardize: Optional[List[int]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run prediction on a given volume.

    This function will automatically choose the correct prediction implementation,
    depending on the model type.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.
        channels_to_standardize: List of channels to standardize. Defaults to None.
        mask: Optional binary mask. If given, the prediction will only be run in
            the foreground region of the mask.

    Returns:
        The predicted volume.
    """
    # make sure either model path or model is passed
    if model is None and model_path is None:
        raise ValueError("Either 'model_path' or 'model' must be provided.")

    if model is not None:
        is_bioimageio = None
    else:
        is_bioimageio = model_path.endswith(".zip")

    if tiling is None:
        tiling = get_default_tiling()

    # We standardize the data for the whole volume beforehand.
    # If we have channels then the standardization is done independently per channel.
    if with_channels:
        input_volume = input_volume.astype(np.float32, copy=False)
        # TODO Check that this is the correct axis.
        if channels_to_standardize is None:  # assume all channels
            channels_to_standardize = range(input_volume.shape[0])
        for ch in channels_to_standardize:
            input_volume[ch] = torch_em.transform.raw.standardize(input_volume[ch])
    else:
        input_volume = torch_em.transform.raw.standardize(input_volume)

    # Run prediction with the bioimage.io library.
    if is_bioimageio:
        if mask is not None:
            raise NotImplementedError
        raise NotImplementedError

    # Run prediction with the torch-em library.
    else:
        if model is None:
            # torch_em expects the root folder of a checkpoint path instead of the checkpoint itself.
            if model_path.endswith("best.pt"):
                model_path = os.path.split(model_path)[0]
        # print(f"tiling {tiling}")
        # Create updated_tiling with the same structure
        updated_tiling = {
            "tile": {},
            "halo": tiling["halo"]  # Keep the halo part unchanged
        }
        # Update tile dimensions
        for dim in tiling["tile"]:
            updated_tiling["tile"][dim] = tiling["tile"][dim] - 2 * tiling["halo"][dim]
        # print(f"updated_tiling {updated_tiling}")
        pred = get_prediction_torch_em(
            input_volume, updated_tiling, model_path, model, verbose, with_channels, mask=mask
        )

    return pred


def get_prediction_torch_em(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.
        mask: Optional binary mask. If given, the prediction will only be run in
            the foreground region of the mask.

    Returns:
        The predicted volume.
    """
    # get block_shape and halo
    block_shape = [tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"]]
    halo = [tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"]]

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suppress warning when loading the model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is None:
            if os.path.isdir(model_path):  # Load the model from a torch_em checkpoint.
                model = torch_em.util.load_model(checkpoint=model_path, device=device)
            else:  # Load the model directly from a serialized pytorch model.
                model = torch.load(model_path)

    # Run prediction with the model.
    with torch.no_grad():

        # Deal with 2D segmentation case
        if len(input_volume.shape) == 2:
            block_shape = [block_shape[1], block_shape[2]]
            halo = [halo[1], halo[2]]

        if mask is not None:
            if verbose:
                print("Run prediction with mask.")
            mask = mask.astype("bool")

        pred = predict_with_halo(
            input_volume, model, gpu_ids=[device],
            block_shape=block_shape, halo=halo,
            preprocess=None, with_channels=with_channels, mask=mask,
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]
        input_root = None
    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
        input_root = input_path

    return input_files, input_root


def _load_input(img_path, extra_files, i):
    # Load the input data.
    if os.path.splitext(img_path)[-1] == ".tif":
        input_volume = imageio.imread(img_path)

    else:
        with open_file(img_path, "r") as f:
            # Try to automatically derive the key with the raw data.
            keys = list(f.keys())
            if len(keys) == 1:
                key = keys[0]
            elif "data" in keys:
                key = "data"
            elif "raw" in keys:
                key = "raw"
            input_volume = f[key][:]

    assert input_volume.ndim in (2, 3)
    # For now we assume this is always tif.
    if extra_files is not None:
        extra_input = imageio.imread(extra_files[i])
        assert extra_input.shape == input_volume.shape
        input_volume = np.stack([input_volume, extra_input], axis=0)

    return input_volume


def _derive_scale(img_path, model_resolution):
    try:
        with mrcfile.open(img_path, "r") as f:
            voxel_size = f.voxel_size
            if len(model_resolution) == 2:
                voxel_size = [voxel_size.y, voxel_size.x]
            else:
                voxel_size = [voxel_size.z, voxel_size.y, voxel_size.x]

        assert len(voxel_size) == len(model_resolution)
        # The voxel size is given in Angstrom and we need to translate it to nanometer.
        voxel_size = [vsize / 10 for vsize in voxel_size]

        # Compute the correct scale factor.
        scale = tuple(vsize / res for vsize, res in zip(voxel_size, model_resolution))
        print("Rescaling the data at", img_path, "by", scale, "to match the training voxel size", model_resolution)

    except Exception:
        warnings.warn(
            f"The voxel size could not be read from the data for {img_path}. "
            "This data will not be scaled for prediction."
        )
        scale = None

    return scale


def inference_helper(
    input_path: str,
    output_root: str,
    segmentation_function: callable,
    data_ext: str = ".mrc",
    extra_input_path: Optional[str] = None,
    extra_input_ext: str = ".tif",
    mask_input_path: Optional[str] = None,
    mask_input_ext: str = ".tif",
    force: bool = False,
    output_key: Optional[str] = None,
    model_resolution: Optional[Tuple[float, float, float]] = None,
    scale: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Helper function to run segmentation for mrc files.

    Args:
        input_path: The path to the input data.
            Can either be a folder. In this case all mrc files below the folder will be segmented.
            Or can be a single mrc file. In this case only this mrc file will be segmented.
        output_root: The path to the output directory where the segmentation results will be saved.
        segmentation_function: The function performing the segmentation.
            This function must take the input_volume as the only argument and must return only the segmentation.
            If you want to pass additional arguments to this function the use 'funtools.partial'
        data_ext: File extension for the image data. By default '.mrc' is used.
        extra_input_path: Filepath to extra inputs that need to be concatenated to the raw data loaded from mrc.
            This enables cristae segmentation with an extra mito channel.
        extra_input_ext: File extension for the extra inputs (by default .tif).
        mask_input_path: Filepath to mask(s) that will be used to restrict the segmentation.
        mask_input_ext: File extension for the mask inputs (by default .tif).
        force: Whether to rerun segmentation for output files that are already present.
        output_key: Output key for the prediction. If none will write an hdf5 file.
        model_resolution: The resolution / voxel size to which the inputs should be scaled for prediction.
            If given, the scaling factor will automatically be determined based on the voxel_size of the input data.
        scale: Fixed factor for scaling the model inputs. Cannot be passed together with 'model_resolution'.
    """
    if (scale is not None) and (model_resolution is not None):
        raise ValueError("You must not provide both 'scale' and 'model_resolution' arguments.")

    # Get the input files. If input_path is a folder then this will load all
    # the mrc files beneath it. Otherwise we assume this is an mrc file already
    # and just return the path to this mrc file.
    input_files, input_root = _get_file_paths(input_path, data_ext)

    # Load extra inputs if the extra_input_path was specified.
    if extra_input_path is None:
        extra_files = None
    else:
        extra_files, _ = _get_file_paths(extra_input_path, extra_input_ext)
        assert len(input_files) == len(extra_files)

    # Load the masks if they were specified.
    if mask_input_path is None:
        mask_files = None
    else:
        mask_files, _ = _get_file_paths(mask_input_path, mask_input_ext)
        assert len(input_files) == len(mask_files)

    for i, img_path in tqdm(enumerate(input_files), total=len(input_files), desc="Processing files"):
        # Determine the output file name.
        input_folder, input_name = os.path.split(img_path)

        if output_key is None:
            fname = os.path.splitext(input_name)[0] + "_prediction.tif"
        else:
            fname = os.path.splitext(input_name)[0] + "_prediction.h5"

        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        # Check if the output path is already present.
        # If it is we skip the prediction, unless force was set to true.
        if os.path.exists(output_path) and not force:
            if output_key is None:
                continue
            else:
                with open_file(output_path, "r") as f:
                    if output_key in f:
                        continue

        # Load the input volume. If we have extra_files then this concatenates the
        # data across a new first axis (= channel axis).
        input_volume = _load_input(img_path, extra_files, i)
        # Load the mask (if given).
        mask = None if mask_files is None else imageio.imread(mask_files[i])

        # Determine the scale factor:
        # If the neither the 'scale' nor 'model_resolution' arguments were passed then set it to None.
        if scale is None and model_resolution is None:
            this_scale = None
        elif scale is not None:   # If 'scale' was passed then use it.
            this_scale = scale
        else:   # Otherwise 'model_resolution' was passed, use it to derive the scaling from the data
            assert model_resolution is not None
            this_scale = _derive_scale(img_path, model_resolution)

        # Run the segmentation.
        segmentation = segmentation_function(input_volume, mask=mask, scale=this_scale)

        # Write the result to tif or h5.
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)

        if output_key is None:
            imageio.imwrite(output_path, segmentation, compression="zlib")
        else:
            with open_file(output_path, "a") as f:
                f.create_dataset(output_key, data=segmentation, compression="gzip")

        print(f"Saved segmentation to {output_path}.")


def get_default_tiling(is_2d: bool = False) -> Dict[str, Dict[str, int]]:
    """Determine the tile shape and halo depending on the available VRAM.

    Args:
        is_2d: Whether to return tiling settings for 2d inference.

    Returns:
        The default tiling settings for the available computational resources.
    """
    if is_2d:
        tile = {"x": 768, "y": 768, "z": 1}
        halo = {"x": 128, "y": 128, "z": 0}
        return {"tile": tile, "halo": halo}

    if torch.cuda.is_available():
        # The default halo size.
        halo = {"x": 64, "y": 64, "z": 16}

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 80}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 64}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 48}
        elif vram >= 10:
            tile = {"x": 256, "y": 256, "z": 32}
            halo = {"x": 64, "y": 64, "z": 8}  # Choose a smaller halo in z.
        else:
            raise NotImplementedError(f"Infererence with a GPU with {vram} GB VRAM is not supported.")

        tiling = {"tile": tile, "halo": halo}
        print(f"Determined tile size for CUDA: {tiling}")

    elif torch.backends.mps.is_available():  # Check for Apple Silicon (MPS)
        tile = {"x": 256, "y": 256, "z": 16}
        halo = {"x": 16, "y": 16, "z": 4}
        tiling = {"tile": tile, "halo": halo}
        print(f"Determined tile size for MPS: {tiling}")

    # I am not sure what is reasonable on a cpu. For now choosing very small tiling.
    # (This will not work well on a CPU in any case.)
    else:
        tiling = {
            "tile": {"x": 96, "y": 96, "z": 16},
            "halo": {"x": 16, "y": 16, "z": 4},
        }
        print(f"Determining default tiling for CPU: {tiling}")

    return tiling


def parse_tiling(
    tile_shape: Tuple[int, int, int],
    halo: Tuple[int, int, int],
    is_2d: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Helper function to parse tiling parameter input from the command line.

    Args:
        tile_shape: The tile shape. If None the default tile shape is used.
        halo: The halo. If None the default halo is used.
        is_2d: Whether to return tiling for a 2d model.

    Returns:
        The tiling specification.
    """

    default_tiling = get_default_tiling(is_2d=is_2d)

    if tile_shape is None:
        tile_shape = default_tiling["tile"]
    else:
        assert len(tile_shape) == 3
        tile_shape = dict(zip("zyx", tile_shape))

    if halo is None:
        halo = default_tiling["halo"]
    else:
        assert len(halo) == 3
        halo = dict(zip("zyx", halo))

    tiling = {"tile": tile_shape, "halo": halo}
    return tiling


#
# Utils for post-processing.
#


def apply_size_filter(
    segmentation: np.ndarray,
    min_size: int,
    verbose: bool = False,
    block_shape: Tuple[int, int, int] = (128, 256, 256),
) -> np.ndarray:
    """Apply size filter to the segmentation to remove small objects.

    Args:
        segmentation: The segmentation.
        min_size: The minimal object size in pixels.
        verbose: Whether to print runtimes.
        block_shape: Block shape for parallelizing the operations.

    Returns:
        The size filtered segmentation.
    """
    if min_size == 0:
        return segmentation
    t0 = time.time()
    if segmentation.ndim == 2 and len(block_shape) == 3:
        block_shape_ = block_shape[1:]
    else:
        block_shape_ = block_shape
    ids, sizes = parallel.unique(segmentation, return_counts=True, block_shape=block_shape_, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return segmentation


def _postprocess_seg_3d(seg, area_threshold=1000, iterations=4, iterations_3d=8):
    # Structure lement for 2d dilation in 3d.
    structure_element = np.ones((3, 3))  # 3x3 structure for XY plane
    structure_3d = np.zeros((1, 3, 3))  # Only applied in the XY plane
    structure_3d[0] = structure_element

    props = regionprops(seg)
    for prop in props:
        # Get bounding box and mask.
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:3], prop.bbox[3:]))
        mask = seg[bb] == prop.label

        # Fill small holes and apply closing.
        mask = remove_small_holes(mask, area_threshold=area_threshold)
        mask = np.logical_or(binary_closing(mask, iterations=iterations), mask)
        mask = np.logical_or(binary_closing(mask, iterations=iterations_3d, structure=structure_3d), mask)
        seg[bb][mask] = prop.label

    return seg


#
# Utils for torch device.
#

def _get_default_device():
    # Check that we're in CI and use the CPU if we are.
    # Otherwise the tests may run out of memory on MAC if MPS is used.
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "cpu"
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        device_type = device if isinstance(device, str) else device.type
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device_type.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device_type.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}. Please choose from 'cpu', 'cuda', or 'mps'.")
    return device
