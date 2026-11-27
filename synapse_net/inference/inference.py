import os
from dataclasses import dataclass
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
# Segmentation of ribbon synapse structures with post-processing.
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


#
# Functions to access SynapseNet's pretrained models.
#


_BIOIMAGEIO_URL = "https://hypha.aicell.io/bioimage-io/artifacts/{}/files/weights.pt?version=v0"
_OWNCLOUD_URL = "https://owncloud.gwdg.de/index.php/s/{}/download"


@dataclass(frozen=True)
class _ModelSpec:
    sha256: str
    url: str
    # The average voxel size of the training data in nm, by axis.
    resolution: Dict[str, float]
    # The training patch shape, by axis. A 2D model has no z axis.
    # The patch shapes are taken from the training scripts, the paths are relative to 'scripts/'.
    # UNCERTAIN marks values that were inferred from related models rather than from the actual training run.
    patch_shape: Dict[str, int]
    segmentation: Callable
    # The overrides of the default U-Net arguments in `_build_unet`, for a model that is downloaded
    # as a state dict from bioimage.io. None for a model that is downloaded as a pickled model.
    unet: Optional[dict] = None


# The plugin offers the subset of these models that is listed in `synapse_net.tools.segmentation_widget`.
_MODELS = {
    "active_zone": _ModelSpec(
        sha256="e953ba83cae6edb24fd74cd6c8ec05ac4f2c406f286b946ea69352d9cbc040b3",
        url=_BIOIMAGEIO_URL.format("loyal-butterfly"),
        resolution={"x": 1.38, "y": 1.38, "z": 1.38},
        patch_shape={"z": 48, "y": 384, "x": 384},  # cooper/revision/train_az.py
        segmentation=segment_active_zone,
        unet={"out_channels": 1},
    ),
    "compartments": _ModelSpec(
        sha256="c4cba43527930e162ce600f71b37ae78d1e76a127de240413086f008bd00cc24",
        url=_BIOIMAGEIO_URL.format("quiet-lobster"),
        resolution={"x": 3.47, "y": 3.47, "z": 3.47},
        patch_shape={"z": 64, "y": 384, "x": 384},  # cooper/training/train_compartments.py
        segmentation=segment_compartments,
        unet={},
    ),
    "mitochondria": _ModelSpec(
        sha256="24625018a5968b36f39fa9d73b121a32e8f66d0f2c0540d3df2e1e39b3d58186",
        url=_OWNCLOUD_URL.format("1T542uvzfuruahD"),
        resolution={"x": 2.07, "y": 2.07, "z": 2.07},
        patch_shape={"z": 32, "y": 256, "x": 256},  # cooper/training/train_mitochondria.py
        segmentation=segment_mitochondria,
    ),
    "mitochondria2": _ModelSpec(
        sha256="45b3e890d928bd9145ecf3b8676f4e1ed5f9938cad1ab51343b9ed07f2d1de24",
        url=_BIOIMAGEIO_URL.format("brave-hedgehog"),
        resolution={"x": 2.87, "y": 2.87, "z": 2.87},
        patch_shape={"z": 32, "y": 256, "x": 256},  # cooper/training/train_mitochondria.py
        segmentation=segment_mitochondria,
        unet={},
    ),
    "mitochondria_fibsem": _ModelSpec(
        sha256="092d94e11753473f4590c4b0aa9a6406b60b81639bdf6a37a1edcffb7476dbca",
        url=_BIOIMAGEIO_URL.format("affable-zebra"),
        resolution={"x": 5.0, "y": 5.0, "z": 25.0},
        patch_shape={"z": 32, "y": 512, "x": 512},  # volume_em/training/train_mitochondria_vol_em.py
        segmentation=segment_mitochondria_fibsem,
        unet={"norm": None, "scale_factors": [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]},
    ),
    "cristae": _ModelSpec(
        sha256="f96c90484f4ea92ac0515a06e389cc117580f02c2aacdc44b5828820cf38c3c3",
        url=_OWNCLOUD_URL.format("Df7OUOyQ1Kc2eEO"),
        resolution={"x": 1.44, "y": 1.44, "z": 1.44},
        patch_shape={"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        segmentation=segment_cristae,
    ),
    "cristae2": _ModelSpec(
        sha256="0864945698862df043adc51c0034289a579b0622a61164e5ebd00a24ee25d075",
        url=_OWNCLOUD_URL.format("qe0R5pRgH2m0pQ5"),
        resolution={"x": 1.44, "y": 1.44, "z": 1.44},
        patch_shape={"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        segmentation=segment_cristae,
    ),
    "cristae3": _ModelSpec(
        sha256="5cb8699487bd21204071cbfb784a7f5c2bb6ab5f347a9d02a913aa27ae70eca4",
        url=_OWNCLOUD_URL.format("51X1et8d7pOfdhU"),
        resolution={"x": 1.44, "y": 1.44, "z": 1.44},
        patch_shape={"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        segmentation=segment_cristae,
    ),
    "cristae4": _ModelSpec(
        sha256="214bf0848c2c3c7459a576c5d46416750e1b9947002674545904f069ddc75066",
        url=_OWNCLOUD_URL.format("XoepbLjeT4csq1t"),
        resolution={"x": 1.74, "y": 1.74, "z": 1.74},
        patch_shape={"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        segmentation=segment_cristae,
    ),
    "cristae5": _ModelSpec(
        sha256="70c13da6687dd994c8a01c30fc5e19613fb55761778db2ae98529cf3a4a55b51",
        url=_BIOIMAGEIO_URL.format("gorgeous-eagle"),
        resolution={"x": 1.74, "y": 1.74, "z": 1.74},
        patch_shape={"z": 32, "y": 256, "x": 256},  # Confirmed by the model's trainer.
        segmentation=segment_cristae,
        unet={"in_channels": 2, "norm": None},
    ),
    "ribbon": _ModelSpec(
        sha256="22b89cf25bdca3477612c403f1c8402b61e8e17f93fa3b6ba988201a0ca076e0",
        url=_BIOIMAGEIO_URL.format("determined-bat"),
        resolution={"x": 1.188, "y": 1.188, "z": 1.188},
        patch_shape={"z": 64, "y": 512, "x": 512},  # inner_ear/training/train_structure_segmentation.py
        segmentation=_segment_ribbon_AZ,
        unet={"out_channels": 3},
    ),
    "vesicles_2d": _ModelSpec(
        sha256="0ddfbcd4cf2ba45f11f854d3bd60726dc1248af186e75fd95934e19a7fdc21a3",
        url=_BIOIMAGEIO_URL.format("adaptable-microbe"),
        resolution={"x": 1.35, "y": 1.35},
        patch_shape={"y": 256, "x": 256},  # cooper/training/train_vesicles.py
        segmentation=segment_vesicles,
        unet={},
    ),
    "vesicles_3d": _ModelSpec(
        sha256="c831e0370607a8010ae33e6b63db291b437ba6cca3ef759e31f49a9e5b871fff",
        url=_BIOIMAGEIO_URL.format("funny-bug"),
        resolution={"x": 1.35, "y": 1.35, "z": 1.35},
        patch_shape={"z": 48, "y": 256, "x": 256},  # cooper/training/train_vesicles.py
        segmentation=segment_vesicles,
        unet={},
    ),
    "vesicles_cryo": _ModelSpec(
        sha256="8b5a4678da3984346e595f3245b45ac63e1d6dd7722b4cd041dd2b145d5ea548",
        url=_BIOIMAGEIO_URL.format("generous-lobster"),
        resolution={"x": 1.35, "y": 1.35, "z": 0.88},
        patch_shape={"z": 48, "y": 256, "x": 256},  # cryo/vesicles/train_domain_adaptation.py
        segmentation=segment_vesicles,
        unet={},
    ),
    # TODO add the correct resolutions for the next three models, these are the resolutions of the source models.
    "vesicles_2d_maus": _ModelSpec(
        sha256="99f70f1ba300f7d0b6468f07457dfcae0250917775d01f1b09b317b6c2e55c96",
        url=_BIOIMAGEIO_URL.format("hardworking-parrot"),
        resolution={"x": 1.35, "y": 1.35},
        patch_shape={"y": 256, "x": 256},  # UNCERTAIN: assumed to be the same as for vesicles_2d.
        segmentation=segment_vesicles,
        unet={},
    ),
    "vesicles_3d_endbulb": _ModelSpec(
        sha256="6ab8a086099b4b16f497a2e20728700c3f9d2437780e38df38edf9aac23b938d",
        url=_BIOIMAGEIO_URL.format("angry-flamingo"),
        resolution={"x": 1.35, "y": 1.35, "z": 1.35},
        patch_shape={"z": 48, "y": 256, "x": 256},  # wichmann/train_domain_adaptation.py
        segmentation=segment_vesicles,
        unet={},
    ),
    "vesicles_3d_innerear": _ModelSpec(
        sha256="679615657484c1c93c1e241cd328943fead2a907e17259853ed5fe8166110696",
        url=_BIOIMAGEIO_URL.format("discreet-angelfish"),
        resolution={"x": 1.35, "y": 1.35, "z": 1.35},
        patch_shape={"z": 48, "y": 256, "x": 256},  # inner_ear/training/train_domain_adaptation_vesicles.py
        segmentation=segment_vesicles,
        unet={},
    ),
}


def _get_model_registry():
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry={name: spec.sha256 for name, spec in _MODELS.items()},
        urls={name: spec.url for name, spec in _MODELS.items()},
    )
    return models


def _build_unet(model_type):
    spec = _MODELS[model_type]
    kwargs = {"in_channels": 1, "out_channels": 2, "initial_features": 32, "gain": 2, "final_activation": "Sigmoid"}
    if "z" not in spec.patch_shape:
        return UNet2d(**{**kwargs, "depth": 4, **spec.unet})
    kwargs.update(scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], norm="InstanceNorm")
    return AnisotropicUNet(**{**kwargs, **spec.unet})


def _state_dict_to_model(state_dict_path, model_type, model_path):
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model = _build_unet(model_type)
    model.load_state_dict({k.removeprefix("unet."): v for k, v in state.items()})
    # Save to a temporary file first, so that an interrupted conversion does not leave a broken model.
    tmp_path = model_path + ".tmp"
    torch.save(model, tmp_path)
    os.replace(tmp_path, model_path)


def get_available_models() -> List[str]:
    """Get the names of all available pretrained models.

    Returns:
        The list of available model names.
    """
    return list(_MODELS)


def get_model_path(model_type: str) -> str:
    """Get the local path to a pretrained model.

    Args:
        The model type.

    Returns:
        The local path to the model.
    """
    model_registry = _get_model_registry()
    spec = _MODELS.get(model_type)
    if spec is None or spec.unet is None:
        return model_registry.fetch(model_type)

    # A bioimage.io model ships a state dict. It is converted once into a pickled model, like the other
    # registry entries, and then removed. The hash in the file name makes a registry update trigger a new download.
    model_path = os.path.join(model_registry.abspath, f"{model_type}-{spec.sha256[:8]}.pt")
    if not os.path.exists(model_path):
        state_dict_path = model_registry.fetch(model_type)
        _state_dict_to_model(state_dict_path, model_type, model_path)
        os.remove(state_dict_path)
    return model_path


def get_model(model_type: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """Get a pretrained model.

    Args:
        model_type: The name of the pretrained model, see `get_available_models`.
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
    return dict(_MODELS[model_type].resolution)


def get_model_training_patch_shape(model_type: str) -> Dict[str, int]:
    """Get the patch shape that was used for training a given pretrained model.

    Args:
        model_type: The name of the pretrained model.

    Returns:
        Mapping of axis (z, y, x) to the patch size of that axis. Only y and x for 2d models.
    """
    return dict(_MODELS[model_type].patch_shape)


def compute_scale_from_voxel_size(
    voxel_size: Dict[str, float],
    model_type: str
) -> List[float]:
    """Compute the appropriate scale factor for inference with a given pretrained model.

    Args:
        voxel_size: The voxel size of the data for inference.
        model_type: The name of the pretrained model.

    Returns:
        The scale factor, as a list in zyx order, or in yx order for 2D data or a 2D model.
    """
    training_voxel_size = get_model_training_resolution(model_type)
    axes = "zyx" if len(voxel_size) == 3 and len(training_voxel_size) == 3 else "yx"
    return [voxel_size[ax] / training_voxel_size[ax] for ax in axes]


#
# Convenience functions for segmentation.
#


def get_segmentation_function(model_type: str) -> Callable:
    """Get the segmentation function associated with a model type.

    Args:
        model_type: The name of the pretrained model.

    Returns:
        The segmentation function used for the model type.

    Raises:
        ValueError: If the model type is unknown.
    """
    if model_type in _MODELS:
        return _MODELS[model_type].segmentation
    # A name that is not in the model registry is resolved by the structure in the name.
    if model_type.startswith("vesicles"):
        return segment_vesicles
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
