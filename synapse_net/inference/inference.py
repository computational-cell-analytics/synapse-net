import os
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
import numpy as np
import pooch
from torch_em.model import AnisotropicUNet, UNet2d

from .active_zone import segment_active_zone
from .compartments import segment_compartments
from .mitochondria import segment_mitochondria, segment_mitochondria_fibsem
from .ribbon_synapse import segment_ribbon_synapse_structures
from .vesicles import segment_vesicles
from .cristae import segment_cristae
from .util import _restore_legacy_attributes, get_default_tiling, get_device
from ..file_utils import get_cache_dir


#
# Functions to access SynapseNet's pretrained models.
#


_BIOIMAGEIO_URL = "https://hypha.aicell.io/bioimage-io/artifacts/{}/files/weights.pt?version=v0"


def _get_model_registry():
    registry = {
        "active_zone": "e953ba83cae6edb24fd74cd6c8ec05ac4f2c406f286b946ea69352d9cbc040b3",
        "compartments": "c4cba43527930e162ce600f71b37ae78d1e76a127de240413086f008bd00cc24",
        "mitochondria": "24625018a5968b36f39fa9d73b121a32e8f66d0f2c0540d3df2e1e39b3d58186",
        "mitochondria_fibsem": "092d94e11753473f4590c4b0aa9a6406b60b81639bdf6a37a1edcffb7476dbca",
        "mitochondria2": "45b3e890d928bd9145ecf3b8676f4e1ed5f9938cad1ab51343b9ed07f2d1de24",
        "cristae": "f96c90484f4ea92ac0515a06e389cc117580f02c2aacdc44b5828820cf38c3c3",
        "cristae2": "0864945698862df043adc51c0034289a579b0622a61164e5ebd00a24ee25d075",
        "cristae3": "5cb8699487bd21204071cbfb784a7f5c2bb6ab5f347a9d02a913aa27ae70eca4",
        "cristae4": "214bf0848c2c3c7459a576c5d46416750e1b9947002674545904f069ddc75066",
        "cristae5": "70c13da6687dd994c8a01c30fc5e19613fb55761778db2ae98529cf3a4a55b51",
        "ribbon": "22b89cf25bdca3477612c403f1c8402b61e8e17f93fa3b6ba988201a0ca076e0",
        "vesicles_2d": "0ddfbcd4cf2ba45f11f854d3bd60726dc1248af186e75fd95934e19a7fdc21a3",
        "vesicles_3d": "c831e0370607a8010ae33e6b63db291b437ba6cca3ef759e31f49a9e5b871fff",
        "vesicles_cryo": "8b5a4678da3984346e595f3245b45ac63e1d6dd7722b4cd041dd2b145d5ea548",
        # Additional models that are only available in the CLI, not in the plugin model selection.
        "vesicles_2d_maus": "99f70f1ba300f7d0b6468f07457dfcae0250917775d01f1b09b317b6c2e55c96",
        "vesicles_3d_endbulb": "6ab8a086099b4b16f497a2e20728700c3f9d2437780e38df38edf9aac23b938d",
        "vesicles_3d_innerear": "679615657484c1c93c1e241cd328943fead2a907e17259853ed5fe8166110696",
    }
    urls = {
        "active_zone": _BIOIMAGEIO_URL.format("loyal-butterfly"),
        "compartments": _BIOIMAGEIO_URL.format("quiet-lobster"),
        "mitochondria": "https://owncloud.gwdg.de/index.php/s/1T542uvzfuruahD/download",
        "mitochondria2": _BIOIMAGEIO_URL.format("brave-hedgehog"),
        "mitochondria_fibsem": _BIOIMAGEIO_URL.format("affable-zebra"),
        "cristae": "https://owncloud.gwdg.de/index.php/s/Df7OUOyQ1Kc2eEO/download",
        "cristae2": "https://owncloud.gwdg.de/index.php/s/qe0R5pRgH2m0pQ5/download",
        "cristae3": "https://owncloud.gwdg.de/index.php/s/51X1et8d7pOfdhU/download",
        "cristae4": "https://owncloud.gwdg.de/index.php/s/XoepbLjeT4csq1t/download",
        "cristae5": _BIOIMAGEIO_URL.format("gorgeous-eagle"),
        "ribbon": _BIOIMAGEIO_URL.format("determined-bat"),
        "vesicles_2d": _BIOIMAGEIO_URL.format("adaptable-microbe"),
        "vesicles_3d": _BIOIMAGEIO_URL.format("funny-bug"),
        "vesicles_cryo": _BIOIMAGEIO_URL.format("generous-lobster"),
        # Additional models that are only available in the CLI, not in the plugin model selection.
        "vesicles_2d_maus": _BIOIMAGEIO_URL.format("hardworking-parrot"),
        "vesicles_3d_endbulb": _BIOIMAGEIO_URL.format("angry-flamingo"),
        "vesicles_3d_innerear": _BIOIMAGEIO_URL.format("discreet-angelfish"),
    }
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def _build_unet(model_type):
    if model_type in ("vesicles_2d", "vesicles_2d_maus"):
        return UNet2d(in_channels=1, out_channels=2, depth=4, initial_features=32, gain=2, final_activation="Sigmoid")
    in_channels, norm, scale_factors = 1, "InstanceNorm", [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    if model_type == "cristae5":
        in_channels, norm = 2, None
    elif model_type == "mitochondria_fibsem":
        norm, scale_factors = None, [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
    out_channels = {"active_zone": 1, "ribbon": 3}.get(model_type, 2)
    return AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors,
        initial_features=32, gain=2, final_activation="Sigmoid", norm=norm,
    )


def _state_dict_to_model(fname, action, pup, model_type):
    # bioimage.io models ship a state dict; convert it once into a pickled model like the other registry entries.
    out = fname + ".pt"
    if action == "fetch" and os.path.exists(out):
        return out
    state = torch.load(fname, map_location="cpu", weights_only=True)
    model = _build_unet(model_type)
    model.load_state_dict({k.removeprefix("unet."): v for k, v in state.items()})
    torch.save(model, out)
    return out


def get_available_models() -> List[str]:
    """Get the names of all available pretrained models.

    Returns:
        The list of available model names.
    """
    model_registry = _get_model_registry()
    return list(model_registry.urls.keys())


def get_model_path(model_type: str) -> str:
    """Get the local path to a pretrained model.

    Args:
        The model type.

    Returns:
        The local path to the model.
    """
    model_registry = _get_model_registry()
    is_bioimageio = "hypha.aicell.io" in model_registry.get_url(model_type)
    processor = partial(_state_dict_to_model, model_type=model_type) if is_bioimageio else None
    model_path = model_registry.fetch(model_type, processor=processor)
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
    model = _restore_legacy_attributes(torch.load(model_path, weights_only=False))
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
        "active_zone": {"x": 1.38, "y": 1.38, "z": 1.38},
        "compartments": {"x": 3.47, "y": 3.47, "z": 3.47},
        "mitochondria": {"x": 2.07, "y": 2.07, "z": 2.07},
        "mitochondria2": {"x": 2.87, "y": 2.87, "z": 2.87},
        "mitochondria_fibsem": {"x": 5.0, "y": 5.0, "z": 25.0},
        "cristae": {"x": 1.44, "y": 1.44, "z": 1.44},
        "cristae2": {"x": 1.44, "y": 1.44, "z": 1.44},
        "cristae3": {"x": 1.44, "y": 1.44, "z": 1.44},
        "cristae4": {"x": 1.74, "y": 1.74, "z": 1.74},
        "cristae5": {"x": 1.74, "y": 1.74, "z": 1.74},
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


def get_model_training_patch_shape(model_type: str) -> Dict[str, int]:
    """Get the patch shape that was used for training a given pretrained model.

    Args:
        model_type: The name of the pretrained model.

    Returns:
        Mapping of axis (z, y, x) to the patch size of that axis. Only y and x for 2d models.
    """
    # The patch shapes are taken from the training scripts, the paths are relative to 'scripts/'.
    # UNCERTAIN marks values that were inferred from related models rather than from the actual training run.
    patch_shapes = {
        "active_zone": {"z": 48, "y": 384, "x": 384},  # cooper/revision/train_az.py
        "compartments": {"z": 64, "y": 384, "x": 384},  # cooper/training/train_compartments.py
        "mitochondria": {"z": 32, "y": 256, "x": 256},  # cooper/training/train_mitochondria.py
        "mitochondria2": {"z": 32, "y": 256, "x": 256},  # cooper/training/train_mitochondria.py
        "mitochondria_fibsem": {"z": 32, "y": 512, "x": 512},  # volume_em/training/train_mitochondria_vol_em.py
        "cristae": {"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        "cristae2": {"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        "cristae3": {"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        "cristae4": {"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        "cristae5": {"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        "ribbon": {"z": 64, "y": 512, "x": 512},  # inner_ear/training/train_structure_segmentation.py
        "vesicles_2d": {"y": 256, "x": 256},  # cooper/training/train_vesicles.py
        "vesicles_3d": {"z": 48, "y": 256, "x": 256},  # cooper/training/train_vesicles.py
        "vesicles_cryo": {"z": 48, "y": 256, "x": 256},  # cryo/vesicles/train_domain_adaptation.py
        "vesicles_2d_maus": {"y": 256, "x": 256},  # UNCERTAIN: assumed to be the same as for vesicles_2d.
        "vesicles_3d_endbulb": {"z": 48, "y": 256, "x": 256},  # wichmann/train_domain_adaptation.py
        "vesicles_3d_innerear": {"z": 48, "y": 256, "x": 256},  # inner_ear/training/train_domain_adaptation_vesicles.py
    }
    return patch_shapes[model_type]


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


def _ribbon_AZ_postprocessing(predictions, vesicles, n_slices_exclude, n_ribbons, resolution, min_membrane_size):
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
        resolution=resolution, min_size=min_membrane_size,
    )

    segmentations = {"ribbon": ribbon, "PD": PD, "membrane": membrane}
    return segmentations


def _segment_ribbon_AZ(
    image,
    model,
    tiling,
    scale,
    verbose,
    return_predictions=False,
    threshold=0.5,
    n_slices_exclude=20,
    min_membrane_size=0,
    n_ribbons=1,
    **kwargs,
):
    # Parse additional keyword arguments from the kwargs.
    vesicles = kwargs.pop("extra_segmentation")
    resolution = kwargs.pop("resolution", None)

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
        segmentations = _ribbon_AZ_postprocessing(
            predictions, vesicles, n_slices_exclude, n_ribbons, resolution, min_membrane_size
        )

    if return_predictions:
        return segmentations, predictions
    return segmentations


def get_segmentation_function(model_type: str) -> Callable:
    """Get the segmentation function associated with a model type.

    Args:
        model_type: The name of the pretrained model.

    Returns:
        The segmentation function used for the model type.

    Raises:
        ValueError: If the model type is unknown.
    """
    if model_type.startswith("vesicles"):
        return segment_vesicles
    if model_type in ("mitochondria", "mitochondria2"):
        return segment_mitochondria
    if model_type == "mitochondria_fibsem":
        return segment_mitochondria_fibsem
    if model_type == "active_zone":
        return segment_active_zone
    if model_type == "compartments":
        return segment_compartments
    if model_type == "ribbon":
        return _segment_ribbon_AZ
    if "cristae" in model_type:
        return segment_cristae
    raise ValueError(f"Unknown model type: {model_type}")


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
        tiling: The tiling settings for inference. By default, the tiling is derived from the patch shape
            that was used for training the model.
        scale: A scale factor for resizing the input before applying the model.
            The output will be scaled back to the initial size.
        verbose: Whether to print detailed information about the prediction and segmentation.
        kwargs: Optional parameters for the segmentation function.

    Returns:
        The segmentation. For models that return multiple segmentations, this function returns a dictionary.
    """
    if tiling is None:
        tiling = get_default_tiling(patch_shape=get_model_training_patch_shape(model_type))
    segmentation_function = get_segmentation_function(model_type)
    if segmentation_function is segment_cristae:
        training_resolution = get_model_training_resolution(model_type)
        voxel_size = np.mean(list(training_resolution.values()))
        segmentation = segmentation_function(
            image, model=model, tiling=tiling, scale=scale, verbose=verbose, voxel_size=voxel_size, **kwargs
        )
    else:
        segmentation = segmentation_function(
            image, model=model, tiling=tiling, scale=scale, verbose=verbose, **kwargs
        )
    return segmentation
