"""Training of the mitochondria model for FIB-SEM volume EM (`mitochondria_fibsem`)."""

import os
from glob import glob
from typing import List, Mapping, Optional, Sequence, Tuple

from torch_em.data import MinInstanceSampler

from .supervised_training import _random_split, _read_split, supervised_training
from .transform import RemoveWhitePatchesAndNormalize

VOL_EM_SCALE_FACTORS = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
"""The U-Net downscaling factors: two anisotropic levels for the 5:1 anisotropy of the data."""


def get_vol_em_mitochondria_paths(
    data_roots: Optional[Mapping[str, str]] = None,
    split_file: Optional[str] = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Get the training and validation data for volume EM mitochondria training.

    Args:
        data_roots: The folders with the training data, by root name. The hdf5 files are found recursively in
            them. If a `split_file` is given, they override the roots stored in it.
        split_file: A json file with the keys 'train' and 'val', and optionally 'roots', whose entries have the
            form '<root name>/<relative path>'. By default the files are split randomly.
        val_fraction: The fraction of the files used for validation in a random split.
        seed: The seed for the random split.

    Returns:
        The filepaths for training.
        The filepaths for validation.
    """
    if split_file is None:
        paths = [path for root in (data_roots or {}).values()
                 for path in glob(os.path.join(root, "**", "*.h5"), recursive=True)]
        return _random_split(paths, val_fraction, seed)

    train_paths, val_paths = (_read_split(split_file, data_roots, key) for key in ("train", "val"))
    missing = [path for path in train_paths + val_paths if not os.path.exists(path)]
    if missing:
        raise ValueError(f"{len(missing)} files from {split_file} do not exist, e.g. {missing[0]}.")
    return train_paths, val_paths


def vol_em_mitochondria_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    patch_shape: Tuple[int, int, int] = (32, 512, 512),
    batch_size: int = 4,
    lr: float = 1e-4,
    n_iterations: int = int(5e4),
    early_stopping: int = 20,
    mixed_precision: bool = False,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    check: bool = False,
) -> None:
    """Train the mitochondria model for FIB-SEM volume EM, with the recipe of `mitochondria_fibsem`.

    The model is an anisotropic U-Net without normalization layers that predicts foreground and boundaries,
    trained at 25 nm in z and 5 nm in xy. The white filler borders of the blocks are removed before the
    percentile normalization; `synapse_net.inference.mitochondria.segment_mitochondria_fibsem` does the same
    at inference. The default batch size and patch shape need an 80 GB GPU.

    Args:
        name: The name of the checkpoint.
        train_paths: The hdf5 files for training, with the data in 'raw' and the mitochondria in
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
        raw_transform=RemoveWhitePatchesAndNormalize(), norm=None, scale_factors=VOL_EM_SCALE_FACTORS,
        early_stopping=early_stopping, mixed_precision=mixed_precision, log_image_interval=50,
        checkpoint_path=checkpoint_path, resume=resume, check=check, num_workers=num_workers, shuffle=False,
    )
