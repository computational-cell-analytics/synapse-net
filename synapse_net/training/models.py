"""Model factories and pretrained backbone utilities for training."""
import warnings
from typing import Tuple
import numpy as np

import torch

from torch_em.model import UNet2d, AnisotropicUNet
from torch_em.model.cryozoo import get_cryozoounet
from torch_em.model.unetr import get_unetr_model
from torch_em.transform.raw import normalize_percentile, get_raw_transform_for_pretrained_vit

from .dataloaders import _determine_ndim

try:
    from micro_sam.v1.util import models as microsam_models
except ImportError:
    microsam_models = None

try:
    from micro_sam.v2.util import models as microsam2_models
except ImportError:
    microsam2_models = None

try:
    from micro_sam.models.vfm import get_vfm_model
except ImportError:
    get_vfm_model = None


def get_3d_model(
    out_channels: int,
    in_channels: int = 1,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 3D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def get_2d_model(
    out_channels: int,
    in_channels: int = 1,
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 2D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        depth=4,
        final_activation=final_activation,
    )
    return model


def _check_backbone(backbone, model_type, patch_shape):
    if backbone == "cryosiam":
        if model_type is not None:
            raise ValueError(f"The 'cryosiam' backbone has no model_type; got model_type='{model_type}'.")
        if _determine_ndim(patch_shape)[0]:
            raise ValueError(f"The 'cryosiam' backbone only supports 3D training; got patch_shape={patch_shape}.")
        if any(s % 8 for s in patch_shape):
            raise ValueError(
                f"The patch_shape for 'cryosiam' backbone must be divisible by 8; got patch_shape={patch_shape}."
            )
    elif (backbone is None) != (model_type is None):
        raise ValueError(
            "backbone and model_type must both be set for ViT backbones; "
            f"got backbone={backbone} and model_type={model_type}."
        )


def _get_model(
    patch_shape, out_channels, in_channels=1, backbone=None, model_type=None,
    encoder_checkpoint=None, final_activation="Sigmoid",
):
    is_2d, ndim = _determine_ndim(patch_shape)

    if backbone == "cryosiam":
        assert in_channels == 1
        return get_cryozoounet(out_channels=out_channels, encoder=backbone, final_activation=final_activation)

    if backbone is not None:
        assert in_channels == 1
        if encoder_checkpoint is None:
            encoder_checkpoint = _get_checkpoint(backbone=backbone, model_type=model_type)
        return get_unetr_model(
            ndim=ndim, backbone=backbone, model_type=model_type,
            encoder_checkpoint=encoder_checkpoint,
            out_channels=out_channels, final_activation=final_activation,
        )

    if is_2d:
        return get_2d_model(out_channels=out_channels, in_channels=in_channels, final_activation=final_activation)

    return get_3d_model(out_channels=out_channels, in_channels=in_channels, final_activation=final_activation)


def normalize_percentile_to_0_1_inverted(raw):
    """Normalize the raw data with percentiles to [0, 1] and invert the contrast."""
    raw = 1 - normalize_percentile(raw)
    return torch.clamp(raw, 0, 1) if torch.is_tensor(raw) else np.clip(raw, 0, 1)


def get_raw_transform(backbone):
    """Get the raw transform and the maximum input value for a pretrained backbone.

    The ViT backbones are handled by `torch_em.transform.raw.get_raw_transform_for_pretrained_vit`.

    Args:
        backbone: The pretrained encoder. Options: "sam", "sam2", "dinov2", "dinov3", "cryosiam", or None.

    Returns:
        The raw transform. It normalizes the raw data with percentiles to [0, 255] for "sam",
            and to [0, 1] for "sam2", "dinov2" and "dinov3". For "cryosiam" it also inverts the contrast.
            None if `backbone` is None, so the loaders use the default torch-em standardization.
        The maximum input value `clip_max`. The intensity augmentations clip their output to
            [0, clip_max], so that the input stays in the expected range. None if `backbone` is None.
    """
    if backbone == "cryosiam":
        return normalize_percentile_to_0_1_inverted, 1

    if backbone not in ("sam", "sam2", "dinov2", "dinov3", None):
        raise ValueError(f"Unsupported backbone '{backbone}'.")
    return get_raw_transform_for_pretrained_vit(backbone)


def _get_checkpoint(backbone, model_type, return_decoder_path=False):

    if backbone == "sam":
        if microsam_models is None:
            raise RuntimeError(
                "The 'sam' backbone requires micro_sam. Install it with 'conda install -c conda-forge micro_sam'."
                )
        model_registry = microsam_models()
        encoder_checkpoint = model_registry.fetch(model_type, progressbar=True)

        if return_decoder_path:
            decoder_name = f"{model_type}_decoder"
            decoder_path = model_registry.fetch(
                decoder_name, progressbar=True
            ) if decoder_name in model_registry.registry else None

    elif backbone == "sam2":
        if microsam2_models is None:
            raise RuntimeError(
                "The 'sam2' backbone requires micro_sam with SAM2 support ('micro_sam.v2') and sam2."
                )
        model_registry = microsam2_models()
        encoder_checkpoint = model_registry.fetch(model_type, progressbar=True)

        if return_decoder_path:
            decoder_name = f"{model_type}_decoder"
            decoder_path = model_registry.fetch(
                decoder_name, progressbar=True
            ) if decoder_name in model_registry.registry else None

    elif backbone == "dinov2":
        if get_vfm_model is None:
            raise RuntimeError("The 'dinov2' backbone requires micro_sam with the 'micro_sam.models.vfm' module.")
        encoder_checkpoint = get_vfm_model(f"{model_type}_dinov2", device="cpu").model.state_dict()
        decoder_path = None

    elif backbone == "dinov3":
        if get_vfm_model is None:
            raise RuntimeError("The 'dinov3' backbone requires micro_sam with the 'micro_sam.models.vfm' module.")
        encoder_checkpoint = get_vfm_model(f"{model_type}_dinov3", device="cpu").model.state_dict()
        decoder_path = None
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    if return_decoder_path:
        return encoder_checkpoint, decoder_path
    else:
        return encoder_checkpoint


def _init_microsam_decoder(model, backbone, model_type, out_channels):
    # Get the micro-sam decoder path
    _, decoder_path = _get_checkpoint(backbone=backbone, model_type=model_type, return_decoder_path=True)

    if decoder_path is None:
        warnings.warn("Oopsie daisy. You choose a `micro-sam` or model without pretrained decoder weights.")
        return model

    # Load the pretrained decoder weights.
    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)

    # Super hacky way of initializing decoder weights - please don't try this at home.
    unetr_state_dict = model.state_dict()
    for k, v in unetr_state_dict.items():
        # If the key starts with encoder, we don't care about it.
        if k.startswith("encoder"):
            continue

        # If the 'out_channels' are different than expected, i.e. 3, we say ciao ciao to it.
        if out_channels != 3 and k.startswith("out_conv"):
            warnings.warn(f"Seems like output channels are different than 'micro-sam'. Hence, we reinitialize '{k}'.")
            unetr_state_dict[k] = v
            continue

        # If it's some weird parameters, we don't care about them too.
        if k not in decoder_state:
            warnings.warn(f"Could not find '{k}' in the pretrained decoder state dict. Hence, we reinitialize '{k}.")
            unetr_state_dict[k] = v
            continue

        # If the parameter shapes mismatch, we don't care about them either.
        if decoder_state[k].shape != v.shape:
            warnings.warn(f"There's a shape mismatch for '{k}'. Hence, we reinitialize '{k}'.")
            unetr_state_dict[k] = v
            continue

        # Keep the expected decoder weights.
        unetr_state_dict[k] = decoder_state[k]

    model.load_state_dict(unetr_state_dict)

    return model
