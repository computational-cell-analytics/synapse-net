"""This submodule implements SynapseNet's segmentation functionality.
"""
from .inference import compute_scale_from_voxel_size, get_model, get_segmentation_function, run_segmentation


__all__ = ["compute_scale_from_voxel_size", "get_model", "get_segmentation_function", "run_segmentation"]
