import os
import re
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pooch

from ..inference.vesicles import segment_vesicles
from ..inference.mitochondria import segment_mitochondria


def load_custom_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    model_path = _clean_filepath(model_path)
    if device is None:
        device = get_device(device)
    try:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except Exception as e:
        print(e)
        print("model path", model_path)
        return None
    return model


def get_model_path(model_type: str) -> str:
    """Get the local path to a given model.

    Args:
        The model type.

    Returns:
        The local path to the model.
    """
    model_registry = get_model_registry()
    model_path = model_registry.fetch(model_type)
    return model_path


def get_model(model_type: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """Get the model for the given segmentation type.

    Args:
        model_type: The model type.
            One of 'vesicles', 'mitochondria', 'active_zone', 'compartments' or 'inner_ear_structures'.
        device: The device to use.

    Returns:
        The model.
    """
    if device is None:
        device = get_device(device)
    model_path = get_model_path(model_type)
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    return model


def run_segmentation(
    image: np.ndarray,
    model: torch.nn.Module,
    model_type: str,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    scale: Optional[List[float]] = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Run synaptic structure segmentation.

    Args:
        image: The input image or image volume.
        model: The segmentation model.
        model_type: The model type. This will determine which segmentation
            post-processing is used.
        tiling: The tiling settings for inference.
        scale: A scale factor for resizing the input before applying the model.
            The output will be scaled back to the initial size.
        verbose: Whether to print detailed information about the prediction and segmentation.
        kwargs: Optional parameter for the segmentation function.

    Returns:
        The segmentation.
    """
    if model_type.startswith("vesicles"):
        segmentation = segment_vesicles(image, model=model, tiling=tiling, scale=scale, verbose=verbose)
    elif model_type == "mitochondria":
        segmentation = segment_mitochondria(image, model=model, tiling=tiling, scale=scale, verbose=verbose)
    elif model_type == "active_zone":
        raise NotImplementedError
    elif model_type == "compartments":
        raise NotImplementedError
    elif model_type == "inner_ear_structures":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return segmentation


def get_cache_dir():
    cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
    return cache_dir


def get_model_training_resolution(model_type):
    resolutions = {
        "active_zone": {"x": 1.44, "y": 1.44, "z": 1.44},
        "compartments": {"x": 3.47, "y": 3.47, "z": 3.47},
        "mitochondria": {"x": 1.0, "y": 1.0, "z": 1.0},  # FIXME: this is a dummy value, we need to determine the real one
        "vesicles_2d": {"x": 1.35, "y": 1.35},
        "vesicles_3d": {"x": 1.35, "y": 1.35, "z": 1.35},
        "vesicles_cryo": {"x": 1.35, "y": 1.35, "z": 0.88},
    }
    return resolutions[model_type]


def get_model_registry():
    registry = {
        "active_zone": "a18f29168aed72edec0f5c2cb1aa9a4baa227812db6082a6538fd38d9f43afb0",
        "compartments": "527983720f9eb215c45c4f4493851fd6551810361eda7b79f185a0d304274ee1",
        "mitochondria": "24625018a5968b36f39fa9d73b121a32e8f66d0f2c0540d3df2e1e39b3d58186",
        "vesicles_2d": "eb0b74f7000a0e6a25b626078e76a9452019f2d1ea6cf2033073656f4f055df1",
        "vesicles_3d": "b329ec1f57f305099c984fbb3d7f6ae4b0ff51ec2fa0fa586df52dad6b84cf29",
        "vesicles_cryo": "782f5a21c3cda82c4e4eaeccc754774d5aaed5929f8496eb018aad7daf91661b",
    }
    urls = {
        "active_zone": "https://owncloud.gwdg.de/index.php/s/zvuY342CyQebPsX/download",
        "compartments": "https://owncloud.gwdg.de/index.php/s/DnFDeTmDDmZrDDX/download",
        "mitochondria": "https://owncloud.gwdg.de/index.php/s/1T542uvzfuruahD/download",
        "vesicles_2d": "https://owncloud.gwdg.de/index.php/s/d72QIvdX6LsgXip/download",
        "vesicles_3d": "https://owncloud.gwdg.de/index.php/s/A425mkAOSqePDhx/download",
        "vesicles_cryo": "https://owncloud.gwdg.de/index.php/s/e2lVdxjCJuZkLJm/download",
    }
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def _get_default_device():
    # check that we're in CI and use the CPU if we are
    # otherwise the tests may run out of memory on MAC if MPS is used.
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
            raise RuntimeError(f"Unsupported device: {device}\n"
                               "Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def get_current_tiling(tiling: dict, default_tiling: dict, model_type: str):
    # get tiling values from qt objects
    for k, v in tiling.items():
        for k2, v2 in v.items():
            if isinstance(v2, int):
                continue
            tiling[k][k2] = v2.value()
    # check if user inputs tiling/halo or not
    if default_tiling == tiling:
        if "2d" in model_type:
            # if its a 2d model expand x,y and set z to 1
            tiling = {
                "tile": {
                    "x": 512,
                    "y": 512,
                    "z": 1
                },
                "halo": {
                    "x": 64,
                    "y": 64,
                    "z": 1
                }
            }
    elif "2d" in model_type:
        # if its a 2d model set z to 1
        tiling["tile"]["z"] = 1
        tiling["halo"]["z"] = 1

    return tiling


def compute_scale_from_voxel_size(
    voxel_size: dict,
    model_type: str
) -> List[float]:
    training_voxel_size = get_model_training_resolution(model_type)
    scale = [
        voxel_size["x"] / training_voxel_size["x"],
        voxel_size["y"] / training_voxel_size["y"],
    ]
    if len(voxel_size) == 3 and len(training_voxel_size) == 3:
        scale.append(
            voxel_size["z"] / training_voxel_size["z"]
        )
    return scale


def _clean_filepath(filepath):
    """
    Cleans a given filepath by:
    - Removing newline characters (\n)
    - Removing escape sequences
    - Stripping the 'file://' prefix if present

    Args:
        filepath (str): The original filepath

    Returns:
        str: The cleaned filepath
    """
    # Remove 'file://' prefix if present
    if filepath.startswith("file://"):
        filepath = filepath[7:]

    # Remove escape sequences and newlines
    filepath = re.sub(r'\\.', '', filepath)
    filepath = filepath.replace('\n', '').replace('\r', '')

    return filepath
