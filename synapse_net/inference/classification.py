"""Classification of segmented instances, e.g. large dense-core vesicles (LDCVs) into their types."""
import json
import os
import pickle
import types
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import center_of_mass

from .util import get_device

CLASSIFICATION_METADATA = "classification_metadata.json"


class _Stub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _StubUnpickler(pickle.Unpickler):
    # Training checkpoints pickle objects from the training code (e.g. the dataset); when that code
    # is not installed, keep only their attributes, which hold the metadata needed for inference.
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Stub,), {})


_stub_pickle = types.SimpleNamespace(Unpickler=_StubUnpickler, load=pickle.load, __name__="stub_pickle")

DEFAULT_NORMALIZATION = {"clip_percentile": 0.01, "eps": 1e-6}


def _build_classification_model(n_classes: int) -> nn.Module:
    from torch_em.model.resnet3d import resnet3d_18

    model = resnet3d_18(in_channels=1, out_channels=n_classes)
    # Small cut-outs: full-resolution stem without the initial downsampling.
    model.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _get_metadata(checkpoint: Dict, checkpoint_path: str) -> Dict:
    # Read from the checkpoint's init, else from the pickled training dataset, else from the json.
    init = checkpoint.get("init", {})
    dataset = init.get("train_dataset")
    metadata_path = os.path.join(os.path.dirname(checkpoint_path), CLASSIFICATION_METADATA)
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    if getattr(dataset, "image_shape", None) is not None:
        raise ValueError(
            f"{checkpoint_path} was trained on cut-outs resized to {dataset.image_shape}, which is not supported."
        )

    label_to_index = getattr(dataset, "label_to_index", None)
    idx_to_label = init.get("idx_to_label") or (
        {idx: label for label, idx in label_to_index.items()} if label_to_index else metadata.get("idx_to_label")
    )
    bbox_size = init.get("bbox_size", getattr(dataset, "bbox_size", metadata.get("bbox_size")))
    halo = init.get("halo", getattr(dataset, "halo", metadata.get("halo", 0)))
    if idx_to_label is None or bbox_size is None:
        raise ValueError(f"Could not find the class names or the cut-out size (bbox_size) in {checkpoint_path}.")

    # Exported checkpoints store it in init; None there means no normalization.
    normalization = init["normalization"] if "normalization" in init else getattr(
        dataset, "normalization", DEFAULT_NORMALIZATION
    )
    if normalization is not None and not isinstance(normalization, dict):
        if type(normalization).__name__ != "CryoETNormalize":
            raise ValueError(f"Unsupported normalization {type(normalization).__name__} in {checkpoint_path}.")
        normalization = {key: getattr(normalization, key) for key in DEFAULT_NORMALIZATION}

    return {
        "bbox_size": int(bbox_size),
        "halo": int(halo),
        "idx_to_label": {int(k): str(v) for k, v in idx_to_label.items()},
        "normalization": normalization,
    }


def load_classification_model(
    checkpoint_path: str, device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, Dict]:
    """Load a classification model and the metadata needed to run it.

    The metadata is read from the checkpoint: from its 'init' entry, or from the training dataset
    stored in it. A 'classification_metadata.json' next to the checkpoint is only used as a fallback.

    Args:
        checkpoint_path: A checkpoint folder containing 'best.pt', or the path to a checkpoint file.
        device: The device to load the model to.

    Returns:
        The model in eval mode.
        The metadata, with the keys 'bbox_size', 'halo', 'idx_to_label' and 'normalization',
            which can be passed to `classify_instances`.
    """
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "best.pt")
    device = get_device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    metadata = _get_metadata(checkpoint, checkpoint_path)

    n_classes = state["fc.weight"].shape[0]
    if sorted(metadata["idx_to_label"]) != list(range(n_classes)):
        raise ValueError(
            f"The model predicts {n_classes} classes, but its idx_to_label has the indices "
            f"{sorted(metadata['idx_to_label'])}."
        )
    model = _build_classification_model(n_classes=n_classes)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, metadata


def export_classification_model(checkpoint_path: str, output_path: str) -> None:
    """Export a training checkpoint to a small checkpoint that only contains what inference needs.

    Training checkpoints also store the optimizer state and the pickled training dataset, which make
    them ~3x larger and contain absolute paths to the training data.

    Args:
        checkpoint_path: A checkpoint folder containing 'best.pt', or the path to a checkpoint file.
        output_path: Where to save the exported checkpoint (.pt), which `load_classification_model` can load.
    """
    model, metadata = load_classification_model(checkpoint_path, device="cpu")
    torch.save({"model_state": model.state_dict(), "init": metadata}, output_path)


def _extract_cube(image: np.ndarray, center: Tuple[int, ...], size: int) -> np.ndarray:
    half = size // 2
    slices, pads = [], []
    for c, extent in zip(center, image.shape):
        lo, hi = c - half, c + half + 1
        slices.append(slice(max(0, lo), min(extent, hi)))
        pads.append((max(0, -lo), max(0, hi - extent)))
    return np.pad(np.asarray(image[tuple(slices)]), pads, mode="constant")


def _normalize(cube: np.ndarray, clip_percentile: Optional[float], eps: float) -> np.ndarray:
    # Same as CryoETNormalize in the training code.
    cube = cube.astype("float32")
    if clip_percentile is not None:
        lo, hi = np.quantile(cube, [clip_percentile, 1.0 - clip_percentile])
        cube = np.clip(cube, lo, hi)
    return ((cube - cube.mean()) / (cube.std(ddof=1) + eps)).astype("float32")


def classify_instances(
    image: np.ndarray,
    segmentation: np.ndarray,
    model: nn.Module,
    bbox_size: int,
    halo: int,
    idx_to_label: Dict[int, str],
    normalization: Optional[Dict] = DEFAULT_NORMALIZATION,
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[int, str]:
    """Classify each instance of a segmentation from a cut-out of the image around its center.

    Args:
        image: The 3D image data.
        segmentation: The 3D instance segmentation, e.g. of LDCVs.
        model: The classification model, see `load_classification_model`.
        bbox_size: The cut-out size the model was trained with.
        halo: The halo the model was trained with; the cut-out edge length is bbox_size + halo, made odd.
        idx_to_label: Mapping from the model's output index to the class name.
        normalization: The 'clip_percentile' and 'eps' of the normalization used in training,
            or None if the model was trained without normalization.
        batch_size: The number of cut-outs that are classified at once.
        device: The device to run the model on.
        progress_callback: Called before the first and after each batch with (number of classified instances,
            total), e.g. to drive a napari progress bar.

    Returns:
        Mapping from each segmentation id to its predicted class name.
    """
    if image.shape != segmentation.shape:
        raise ValueError(f"Image shape {image.shape} does not match segmentation shape {segmentation.shape}.")

    seg_ids = np.unique(segmentation)
    seg_ids = seg_ids[seg_ids != 0].tolist()
    if not seg_ids:
        return {}

    size = bbox_size + halo
    size = size + 1 if size % 2 == 0 else size
    centers = center_of_mass(segmentation, labels=segmentation, index=seg_ids)
    centers = [tuple(int(round(c)) for c in center) for center in centers]

    device = get_device(device)
    model = model.to(device).eval()
    predictions = []
    if progress_callback is not None:
        progress_callback(0, len(centers))
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            cubes = [_extract_cube(image, center, size) for center in centers[start:start + batch_size]]
            if normalization is not None:
                cubes = [_normalize(cube, **normalization) for cube in cubes]
            cubes = np.stack(cubes).astype("float32")
            logits = model(torch.from_numpy(cubes).unsqueeze(1).to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            if progress_callback is not None:
                progress_callback(len(predictions), len(centers))

    return {seg_id: idx_to_label[pred] for seg_id, pred in zip(seg_ids, predictions)}


def get_class_masks(
    segmentation: np.ndarray, seg_id_to_class: Dict[int, str], class_names: List[str]
) -> Dict[str, np.ndarray]:
    """Split a segmentation into one mask per class; each instance keeps its segmentation id.

    Args:
        segmentation: The instance segmentation.
        seg_id_to_class: Mapping from segmentation id to class name, see `classify_instances`.
        class_names: The classes to return a mask for, also if they contain no instance.

    Returns:
        Mapping from class name to its mask.
    """
    max_id = int(segmentation.max()) if segmentation.size else 0
    masks = {}
    for class_name in class_names:
        ids = np.array([seg_id for seg_id, name in seg_id_to_class.items() if name == class_name], dtype="int64")
        lut = np.zeros(max_id + 1, dtype=segmentation.dtype)
        lut[ids] = ids
        masks[class_name] = lut[segmentation]
    return masks


def get_combined_class_mask(
    segmentation: np.ndarray, seg_id_to_class: Dict[int, str], class_names: List[str]
) -> np.ndarray:
    """Combine the classes into one mask, in which each instance is labeled with its class id.

    Args:
        segmentation: The instance segmentation.
        seg_id_to_class: Mapping from segmentation id to class name, see `classify_instances`.
        class_names: The class names; the class at position i gets the label id i + 1.

    Returns:
        The combined mask, with 0 for background.
    """
    class_to_label = {name: label for label, name in enumerate(class_names, start=1)}
    max_id = int(segmentation.max()) if segmentation.size else 0
    lut = np.zeros(max_id + 1, dtype="uint8" if len(class_names) < 256 else "uint16")
    for seg_id, class_name in seg_id_to_class.items():
        lut[seg_id] = class_to_label[class_name]
    return lut[segmentation]
