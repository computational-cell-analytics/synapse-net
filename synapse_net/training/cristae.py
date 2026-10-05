"""Training of the cristae model for electron tomography (`cristae5`)."""

import json
import os
from glob import glob
from typing import List, Mapping, Optional, Sequence, Tuple

import torch_em
from torch_em.data import MinInstanceSampler

from .loss import MaskedDiceLossPerSample
from .supervised_training import _random_split, supervised_training
from .transform import AugmentedMitoStateMaskTransform, standardize_channel


def _read_split(split_file, data_roots, key):
    with open(split_file) as f:
        split = json.load(f)
    # The entries have the form '<root name>/<relative path>'. The roots stored in the split file can be
    # overridden, so that the data can be moved.
    roots = {**split.get("roots", {}), **(data_roots or {})}
    paths = []
    for entry in split.get(key, []):
        root, *parts = entry.split("/")
        if root not in roots:
            raise ValueError(f"{split_file} refers to the unknown data root '{root}'.")
        paths.append(os.path.join(roots[root], *parts))
    return paths


def get_cristae_test_paths(split_file: str, data_roots: Optional[Mapping[str, str]] = None) -> List[str]:
    """Get the test data of a split file for cristae training.

    Args:
        split_file: The json file with the split.
        data_roots: Folders that override the data roots stored in the split file, by root name.

    Returns:
        The filepaths for testing.
    """
    return _read_split(split_file, data_roots, "test")


def get_cristae_paths(
    data_roots: Optional[Mapping[str, str]] = None,
    split_file: Optional[str] = None,
    exclude: Sequence[str] = (),
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Get the training and validation data for cristae training.

    Args:
        data_roots: The folders with the training data, by root name. The '*_combined.h5' files are found
            recursively in them. If a `split_file` is given, they override the roots stored in it.
        split_file: A json file with the keys 'train' and 'val', and optionally 'test' and 'roots', whose
            entries have the form '<root name>/<relative path>'. By default the files are split randomly.
        exclude: Substrings of the filepaths to leave out of a random split.
        val_fraction: The fraction of the files used for validation in a random split.
        seed: The seed for the random split.

    Returns:
        The filepaths for training.
        The filepaths for validation.
    """
    if split_file is None:
        paths = [path for root in (data_roots or {}).values()
                 for path in glob(os.path.join(root, "**", "*_combined.h5"), recursive=True)]
        return _random_split([path for path in paths if not any(ex in path for ex in exclude)], val_fraction, seed)

    train_paths, val_paths = (_read_split(split_file, data_roots, key) for key in ("train", "val"))
    missing = [path for path in train_paths + val_paths if not os.path.exists(path)]
    if missing:
        raise ValueError(f"{len(missing)} files from {split_file} do not exist, e.g. {missing[0]}.")
    return train_paths, val_paths


def cristae_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    patch_shape: Tuple[int, int, int] = (32, 256, 256),
    batch_size: int = 24,
    lr: float = 1.7e-4,
    n_iterations: int = int(1e5),
    early_stopping: int = 25,
    membrane_w_pos: float = 3.0,
    membrane_w_neg: float = 2.0,
    membrane_band_nm: float = 12.0,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    check: bool = False,
) -> None:
    """Train the cristae model for electron tomography, with the recipe of `cristae5`.

    The network gets two input channels, the tomogram and the mitochondria state, which is 0 for background, 1 for
    mitochondria with cristae annotations and 2 for mitochondria without. The mitochondria without annotations are
    excluded from the loss, and the loss is weighted towards the membrane of the annotated mitochondria, which
    improves the detection of cristae junctions. The default batch size and patch shape need an 80 GB GPU.

    Args:
        name: The name of the checkpoint.
        train_paths: The hdf5 files for training, with the tomogram and the mitochondria state in
            'raw_mitos_combined' and the cristae in 'labels/cristae'.
        val_paths: The hdf5 files for validation.
        save_root: The folder for the checkpoint and the logs.
        patch_shape: The patch shape for training.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The maximal number of iterations.
        early_stopping: The number of epochs without improvement after which training is stopped.
        membrane_w_pos: The loss weight for cristae within `membrane_band_nm` of the membrane.
        membrane_w_neg: The loss weight for the other voxels within the band. Pass 1.0 for both weights to
            train without the membrane weighting.
        membrane_band_nm: The thickness of the membrane band in nanometer.
        num_workers: The number of workers for the dataloaders.
        checkpoint_path: A checkpoint to initialize the weights from.
        resume: Whether to continue the previous run with this name, see `supervised_training`.
        check: Whether to check the dataloaders instead of training.
    """
    transform = AugmentedMitoStateMaskTransform(
        torch_em.transform.get_augmentations(3), band_nm=membrane_band_nm, w_pos=membrane_w_pos, w_neg=membrane_w_neg,
    )
    # These are only valid for multi-process loading.
    worker_kwargs = dict(persistent_workers=True, prefetch_factor=4) if num_workers > 0 else {}
    supervised_training(
        name=name, train_paths=tuple(train_paths), val_paths=tuple(val_paths), save_root=save_root,
        raw_key="raw_mitos_combined", label_key="labels/cristae", patch_shape=patch_shape, batch_size=batch_size,
        lr=lr, n_iterations=n_iterations, sampler=MinInstanceSampler(p_reject=0.95), loss_fn=MaskedDiceLossPerSample(),
        in_channels=2, out_channels=2, norm=None, transform=transform, raw_transform=standardize_channel,
        with_channels=True, early_stopping=early_stopping, mixed_precision=True, log_image_interval=50,
        checkpoint_path=checkpoint_path, resume=resume, check=check, num_workers=num_workers, shuffle=False,
        **worker_kwargs,
    )
