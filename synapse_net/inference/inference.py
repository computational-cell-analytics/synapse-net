import os
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pooch

from .active_zone import segment_active_zone
from .compartments import segment_compartments
from .mitochondria import segment_mitochondria
from .ribbon_synapse import segment_ribbon_synapse_structures
from .vesicles import segment_vesicles
from .cristae import segment_cristae
from .util import get_device
from ..file_utils import get_cache_dir


#
# Functions to access SynapseNet's pretrained models.
#


def _get_model_registry():
    registry = {
        "active_zone": "a18f29168aed72edec0f5c2cb1aa9a4baa227812db6082a6538fd38d9f43afb0",
        "compartments": "527983720f9eb215c45c4f4493851fd6551810361eda7b79f185a0d304274ee1",
        "mitochondria": "24625018a5968b36f39fa9d73b121a32e8f66d0f2c0540d3df2e1e39b3d58186",
        "mitochondria2": "553decafaff4838fff6cc8347f22c8db3dee5bcbeffc34ffaec152f8449af673",
        "cristae": "f96c90484f4ea92ac0515a06e389cc117580f02c2aacdc44b5828820cf38c3c3",
        "ribbon": "7c947f0ddfabe51a41d9d05c0a6ca7d6b238f43df2af8fffed5552d09bb075a9",
        "vesicles_2d": "eb0b74f7000a0e6a25b626078e76a9452019f2d1ea6cf2033073656f4f055df1",
        "vesicles_3d": "b329ec1f57f305099c984fbb3d7f6ae4b0ff51ec2fa0fa586df52dad6b84cf29",
        "vesicles_cryo": "782f5a21c3cda82c4e4eaeccc754774d5aaed5929f8496eb018aad7daf91661b",
        # Additional models that are only available in the CLI, not in the plugin model selection.
        "vesicles_2d_maus": "01506895df6343fc33ffc9c9eb3f975bf42eb4eaaaf4848bac83b57f1b46e460",
        "vesicles_3d_endbulb": "8582c7e3e5f16ef2bf34d6f9e34644862ca3c76835c9e7d44475c9dd7891d228",
        "vesicles_3d_innerear": "924f0f7cfb648a3a6931c1d48d8b1fdc6c0c0d2cb3330fe2cae49d13e7c3b69d",
    }
    urls = {
        "active_zone": "https://owncloud.gwdg.de/index.php/s/zvuY342CyQebPsX/download",
        "compartments": "https://owncloud.gwdg.de/index.php/s/DnFDeTmDDmZrDDX/download",
        "mitochondria": "https://owncloud.gwdg.de/index.php/s/1T542uvzfuruahD/download",
        "mitochondria2": "https://owncloud.gwdg.de/index.php/s/GZghrXagc54FFXd/download",
        "cristae": "https://owncloud.gwdg.de/index.php/s/Df7OUOyQ1Kc2eEO/download",
        "ribbon": "https://owncloud.gwdg.de/index.php/s/S3b5l0liPP1XPYA/download",
        "vesicles_2d": "https://owncloud.gwdg.de/index.php/s/d72QIvdX6LsgXip/download",
        "vesicles_3d": "https://owncloud.gwdg.de/index.php/s/A425mkAOSqePDhx/download",
        "vesicles_cryo": "https://owncloud.gwdg.de/index.php/s/e2lVdxjCJuZkLJm/download",
        # Additional models that are only available in the CLI, not in the plugin model selection.
        "vesicles_2d_maus": "https://owncloud.gwdg.de/index.php/s/sZ8woLr0zs5zOpv/download",
        "vesicles_3d_endbulb": "https://owncloud.gwdg.de/index.php/s/16tmnWrEDpYIMzU/download",
        "vesicles_3d_innerear": "https://owncloud.gwdg.de/index.php/s/UFUCYivsCxrqISX/download",
    }
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def get_model_path(model_type: str) -> str:
    """Get the local path to a pretrained model.

    Args:
        The model type.

    Returns:
        The local path to the model.
    """
    model_registry = _get_model_registry()
    model_path = model_registry.fetch(model_type)
    return model_path


def get_model(model_type: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """Get the model for a specific segmentation type.

    Args:
        model_type: The model for one of the following segmentation tasks:
            'vesicles_3d', 'active_zone', 'compartments', 'mitochondria', 'ribbon', 'vesicles_2d', 'vesicles_cryo'.
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


#
# Functions for training resolution / voxel size.
#


def get_model_training_resolution(model_type: str) -> Dict[str, float]:
    """Get the average resolution / voxel size of the training data for a given pretrained model.

    Args:
        model_type: The name of the pretrained model.

    Returns:
        Mapping of axis (x, y, z) to the voxel size (in nm) of that axis.
    """
    resolutions = {
        "active_zone": {"x": 1.44, "y": 1.44, "z": 1.44},
        "compartments": {"x": 3.47, "y": 3.47, "z": 3.47},
        "mitochondria": {"x": 2.07, "y": 2.07, "z": 2.07},
        "ribbon": {"x": 1.188, "y": 1.188, "z": 1.188},
        "vesicles_2d": {"x": 1.35, "y": 1.35},
        "vesicles_3d": {"x": 1.35, "y": 1.35, "z": 1.35},
        "vesicles_cryo": {"x": 1.35, "y": 1.35, "z": 0.88},
        # TODO add the correct resolutions, these are the resolutions of the source models.
        "vesicles_2d_maus": {"x": 1.35, "y": 1.35},
        "vesicles_3d_endbulb": {"x": 1.35, "y": 1.35, "z": 1.35},
        "vesicles_3d_innerear": {"x": 1.35, "y": 1.35, "z": 1.35},
    }
    return resolutions[model_type]


def compute_scale_from_voxel_size(
    voxel_size: Dict[str, float],
    model_type: str
) -> List[float]:
    """Compute the appropriate scale factor for inference with a given pretrained model.

    Args:
        voxel_size: The voxel size of the data for inference.
        model_type: The name of the pretrained model.

    Returns:
        The scale factor, as a list in zyx order.
    """
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


#
# Convenience functions for segmentation.
#


def _ribbon_AZ_postprocessing(predictions, vesicles, n_slices_exclude, n_ribbons):
    from synapse_net.inference.postprocessing import (
        segment_ribbon, segment_presynaptic_density, segment_membrane_distance_based,
    )

    ribbon = segment_ribbon(
        predictions["ribbon"], vesicles, n_slices_exclude=n_slices_exclude, n_ribbons=n_ribbons,
        max_vesicle_distance=40,
    )
    PD = segment_presynaptic_density(
        predictions["PD"], ribbon, n_slices_exclude=n_slices_exclude, max_distance_to_ribbon=40,
    )
    ref_segmentation = PD if PD.sum() > 0 else ribbon
    membrane = segment_membrane_distance_based(
        predictions["membrane"], ref_segmentation, max_distance=500, n_slices_exclude=n_slices_exclude,
    )

    segmentations = {"ribbon": ribbon, "PD": PD, "membrane": membrane}
    return segmentations


def _segment_ribbon_AZ(image, model, tiling, scale, verbose, return_predictions=False, **kwargs):
    # Parse additional keyword arguments from the kwargs.
    vesicles = kwargs.pop("extra_segmentation")
    threshold = kwargs.pop("threshold", 0.5)
    n_slices_exclude = kwargs.pop("n_slices_exclude", 20)
    n_ribbons = kwargs.pop("n_slices_exclude", 1)

    predictions = segment_ribbon_synapse_structures(
        image, model=model, tiling=tiling, scale=scale, verbose=verbose, threshold=threshold, **kwargs
    )

    # Otherwise, just return the predictions.
    if vesicles is None:
        if verbose:
            print("Vesicle segmentation was not passed, WILL NOT run post-processing.")
        segmentations = predictions

    # If the vesicles were passed then run additional post-processing.
    else:
        if verbose:
            print("Vesicle segmentation was passed, WILL run post-processing.")
        segmentations = _ribbon_AZ_postprocessing(predictions, vesicles, n_slices_exclude, n_ribbons)

    if return_predictions:
        return segmentations, predictions
    return segmentations


def run_segmentation(
    image: np.ndarray,
    model: torch.nn.Module,
    model_type: str,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    scale: Optional[List[float]] = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray | Dict[str, np.ndarray]:
    """Run synaptic structure segmentation.

    Args:
        image: The input image or image volume.
        model: The segmentation model.
        model_type: The model type. This will determine which segmentation post-processing is used.
        tiling: The tiling settings for inference.
        scale: A scale factor for resizing the input before applying the model.
            The output will be scaled back to the initial size.
        verbose: Whether to print detailed information about the prediction and segmentation.
        kwargs: Optional parameters for the segmentation function.

    Returns:
        The segmentation. For models that return multiple segmentations, this function returns a dictionary.
    """
    if model_type.startswith("vesicles"):
        segmentation = segment_vesicles(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    elif model_type == "mitochondria" or model_type == "mitochondria2":
        segmentation = segment_mitochondria(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    elif model_type == "active_zone":
        segmentation = segment_active_zone(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    elif model_type == "compartments":
        segmentation = segment_compartments(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    elif model_type == "ribbon":
        segmentation = _segment_ribbon_AZ(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    elif model_type == "cristae":
        segmentation = segment_cristae(image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return segmentation
