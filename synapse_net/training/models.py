"""
Code ported from micro-match.
Author: Marei Freitag
"""
import os
import pooch
import warnings
from typing import Literal, Tuple
from functools import partial
import numpy as np

import torch
import micro_sam
from micro_sam.v2.util import _get_checkpoint as _get_sam2_checkpoint

from torch_em.model.unetr import UNETR2D, UNETR3D
from torch_em.model import UNet2d, AnisotropicUNet
from torch_em.transform.raw import normalize_percentile, normalize

def get_unetr_model(
    ndim: int,
    backbone: Literal["sam", "sam2", "dinov2", "dinov3"],
    model_type: str,
    out_channels: int = 1,
    init_decoder: bool = False,
    final_activation="Sigmoid",
):
    if backbone not in ("sam", "sam2", "dinov2", "dinov3"):
        raise ValueError(f"Unsupported backbone '{backbone}'.")
    
    # Get the model class.
    if ndim == 2:
        model_class = UNETR2D
    else:
        model_class = partial(UNETR3D, use_strip_pooling=False)

    # Load the model
    model = model_class(
        img_size=518 if backbone == "dinov2" else 1024,
        backbone=backbone,
        encoder=_get_vit_type(model_type),
        encoder_checkpoint=_get_checkpoint(backbone=backbone, model_type=model_type),
        out_channels=out_channels,
        resize_input=True,
        final_activation=final_activation,
        use_skip_connection=False,
        use_conv_transpose=False,
        use_sam_stats="sam" in backbone,
        use_dino_stats="dino" in backbone,
        embed_dim=_get_embed_dim(backbone=backbone),
    )
    _get_checkpoint(backbone, model_type, return_decoder_path=False)

    if init_decoder and ndim == 2:
        model = _init_microsam_decoder(model, backbone, model_type, out_channels)
    # Init decoder for microSAM2 model
    elif init_decoder and backbone == "sam2" and model_type == "hvit_t_em_organelles":
        model = _init_microsam_decoder(model, backbone, model_type, out_channels)

    return model

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

def normalize_percentile_to_0_1(raw):
    raw = normalize_percentile(raw)
    if torch.is_tensor(raw):
        return torch.clamp(raw, 0, 1)
    raw = np.clip(raw, 0, 1)
    return raw


def normalize_to_0_1(raw):
    return normalize_percentile_to_0_1(raw)


def normalize_percentile_to_0_255(raw):
    raw = normalize_percentile_to_0_1(raw)
    raw = raw * 255.0
    return raw


def normalize_to_0_255(raw):
    raw = normalize(raw)
    raw = raw * 255.0
    return raw


def get_raw_transform(backbone):
    if backbone in ("sam2", "dinov2", "dinov3"):
        raw_transform = normalize_percentile_to_0_1
        clip_max = 1
    elif backbone is not None:
        raw_transform = normalize_percentile_to_0_255
        clip_max = 255
    else:
        raw_transform = normalize_percentile_to_0_1
        clip_max = 1
    return (raw_transform, clip_max)


def _get_dinov2_checkpoint(model_type: str):
    """Downloads the model checkpoints and returns filepath to it.
    """
    assert isinstance(model_type, str), "Watch out, we need the 'model_type' name."

    # Let's create a cache directory.
    save_directory = os.path.expanduser(pooch.os_cache("micro_match/dinov2_models"))

    # The weights are stored on owncloud for now
    urls = {"vit_b": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"}
    hashes = {"vit_b": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"}

    # Make an explicit check
    assert model_type in urls, "Oops, seems like the specific DINOv2 model is not worked out atm."

    # Download the model weights
    pooch.retrieve(
        url=urls[model_type],
        known_hash=hashes[model_type],
        fname=model_type,
        path=save_directory,
        progressbar=True,
    )

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, model_type)

    return checkpoint_path


def _get_dinov3_checkpoint(model_type: str):
    """Downloads the model checkpoint and returns filepath to it.
    """
    assert isinstance(model_type, str), "Watch out, we need the 'model_type' name."

    # Let's create a cache directory.
    save_directory = os.path.expanduser(pooch.os_cache("micro_match/dinov3_models"))

    # The weights are stored on owncloud for now
    urls = {"vit_b": "https://owncloud.gwdg.de/index.php/s/PvTzG3fdo2pnNCz/download"}
    hashes = {"vit_b": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"}

    # Make an explicit check
    assert model_type in urls, "Oops, seems like the specific DINOv3 model is not stored on OwnCloud."

    # Download the model weights
    pooch.retrieve(
        url=urls[model_type],
        known_hash=hashes[model_type],
        fname=model_type,
        path=save_directory,
        progressbar=True,
    )

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, model_type)

    return checkpoint_path


def _get_microsam2_checkpoint(model_type: str, cache_dir=None):
    """Download MicroSAM2 weights and split into encoder/decoder checkpoints.

    Args:
        cache_dir: Directory to cache the weights. Defaults to the micro_sam
            pooch cache (``~/.cache/micro_sam/v2/models`` or OS equivalent).

    Returns:
        Absolute path to the cached model file.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(pooch.os_cache("micro_match/microsam2_models"))

    urls = {"hvit_t_em_organelles": "https://owncloud.gwdg.de/index.php/s/kMxqsRL1FG9pslC/download"}
    hashes = {"hvit_t_em_organelles": "a34b4a4e9360d48eafe7995438610bf656325acae71cd1f3f512e236511b212e"}

    full_checkpoint = pooch.retrieve(
        url=urls[model_type],
        known_hash=hashes[model_type],
        fname=model_type,
        path=cache_dir,
        progressbar=True,
    )

    encoder_path = os.path.join(cache_dir, f"{model_type}_encoder.pt")
    decoder_path = os.path.join(cache_dir, f"{model_type}_decoder.pt")

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        state = torch.load(
            full_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        decoder_state = {
            k: v
            for k, v in state.items()
            if (
                not k.startswith("encoder.")
                and not k.startswith("out_conv")
            )
        }

        torch.save(encoder_state, encoder_path)
        torch.save(decoder_state, decoder_path)

    return encoder_path, decoder_path


def _get_checkpoint(backbone, model_type, return_decoder_path=False):

    if backbone == "sam":
        checkpoint_path, _, decoder_path = micro_sam.util._download_sam_model(model_type)

    elif backbone == "sam2":
        if model_type == 'hvit_t_em_organelles':  # Not yet in the zoo
            checkpoint_path, decoder_path = _get_microsam2_checkpoint(model_type)
        else:
            checkpoint_path = _get_sam2_checkpoint(model_type)

    elif backbone == "dinov2":
        checkpoint_path = _get_dinov2_checkpoint(model_type)

    elif backbone == "dinov3":
        checkpoint_path = _get_dinov3_checkpoint(model_type)
    # elif backbone == "sam3":
    #     assert model_type == "vit_pe"
    #     import micro_sam3
    #     checkpoint_path = micro_sam3.util._get_checkpoint()
    else:
        raise ValueError()

    if return_decoder_path:
        return checkpoint_path, decoder_path
    else:
        return checkpoint_path


def _get_embed_dim(backbone):
    embed_dim = None

    if backbone == "sam2":
        embed_dim = 256
    elif backbone in ["sam", "dinov2", "dinov3"]:
        embed_dim = 768
    elif backbone == "sam3":
        embed_dim = 1024

    return embed_dim


def _get_vit_type(model_type: str) -> str:
    if model_type is None:
        return model_type
    else:
        parts = model_type.split("_")
        model_type = "_".join(parts[:2])
        return model_type


def _init_microsam_decoder(model, backbone, model_type, out_channels):
    # Get the micro-sam decoder path
    _, decoder_path = _get_checkpoint(backbone=backbone, model_type=model_type, return_decoder_path=True)

    if decoder_path is None:
        warnings.warn("Oopsie daisy. You choose a `micro-sam` or model without pretrained decoder weights.")
        return model

    # Load the pretrained decoder weights.
    decoder_state = torch.load(decoder_path, map_location="cpu")

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