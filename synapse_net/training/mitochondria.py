"""Training of the mitochondria model for electron tomography (`mitochondria2`)."""

import json
import os
from glob import glob
from typing import List, Optional, Sequence, Tuple

import torch_em
from torch_em.data import MinInstanceSampler

from .supervised_training import _random_split, supervised_training


def get_mitochondria_paths(
    data_root: str, split_file: Optional[str] = None, val_fraction: float = 0.15, seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Get the training and validation data for mitochondria training.

    Args:
        data_root: The folder with the training data, hdf5 files that are found recursively.
        split_file: A json file with the keys 'train' and 'val', holding filepaths relative to `data_root`.
            By default the files are split randomly.
        val_fraction: The fraction of the files used for validation in a random split.
        seed: The seed for the random split.

    Returns:
        The filepaths for training.
        The filepaths for validation.
    """
    if split_file is None:
        return _random_split(glob(os.path.join(data_root, "**", "*.h5"), recursive=True), val_fraction, seed)

    with open(split_file) as f:
        split = json.load(f)
    # The entries use '/', join their parts so that the paths match the ones found by glob on Windows.
    train_paths, val_paths = (
        [os.path.join(data_root, *entry.split("/")) for entry in split[key]] for key in ("train", "val")
    )
    missing = [path for path in train_paths + val_paths if not os.path.exists(path)]
    if missing:
        raise ValueError(f"{len(missing)} files from {split_file} do not exist, e.g. {missing[0]}.")
    return train_paths, val_paths


def mitochondria_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    patch_shape: Tuple[int, int, int] = (32, 256, 256),
    batch_size: int = 8,
    lr: float = 1e-4,
    n_iterations: int = int(1.5e5),
    early_stopping: int = 20,
    mixed_precision: bool = False,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    check: bool = False,
) -> None:
    """Train the mitochondria model for electron tomography, with the recipe of `mitochondria2`.

    The model predicts foreground and boundaries from tomograms that are normalized with the 1st and 99th
    percentile. This normalization has to be applied at inference too, by passing
    `preprocess=torch_em.transform.raw.normalize_percentile` to `segment_mitochondria`.
    The default batch size and patch shape need an 80 GB GPU; `mixed_precision` reduces the memory.

    Args:
        name: The name of the checkpoint.
        train_paths: The hdf5 files for training, with the tomogram in 'raw' and the mitochondria in
            'labels/mitochondria'.
        val_paths: The hdf5 files for validation.
        save_root: The folder for the checkpoint and the logs.
        patch_shape: The patch shape for training.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The maximal number of iterations.
        early_stopping: The number of epochs without improvement after which training is stopped.
        mixed_precision: Whether to train with mixed precision. The published model was trained without.
        num_workers: The number of workers for the dataloaders.
        checkpoint_path: A checkpoint to initialize the weights from.
        resume: Whether to continue the previous run with this name, see `supervised_training`.
        check: Whether to check the dataloaders instead of training.
    """
    supervised_training(
        name=name, train_paths=tuple(train_paths), val_paths=tuple(val_paths), save_root=save_root,
        raw_key="raw", label_key="labels/mitochondria", patch_shape=patch_shape, batch_size=batch_size, lr=lr,
        n_iterations=n_iterations, n_samples_train=500, n_samples_val=500, sampler=MinInstanceSampler(p_reject=0.95),
        raw_transform=torch_em.transform.raw.normalize_percentile, early_stopping=early_stopping,
        mixed_precision=mixed_precision, log_image_interval=50, checkpoint_path=checkpoint_path, resume=resume,
        check=check, num_workers=num_workers, shuffle=False,
    )
