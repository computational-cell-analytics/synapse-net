"""Training for cristae segmentation in electron tomography.

This module reproduces the recipe that was used to train SynapseNet's `cristae5` model, which
segments cristae in electron tomograms at a resolution of 1.74 nm.

Cristae training differs from the other training functions in this package in three ways:

- The network sees **two input channels**: the tomogram and a semantic mitochondria state, where 0 is
  background, 1 is a mitochondrion that carries cristae annotations and 2 is a mitochondrion that
  does not. Both are stored in a single hdf5 dataset of shape (2, z, y, x).
- The voxels of unannotated mitochondria (state 2) are **excluded from the loss**, so that the network
  is not penalized for predicting cristae where no annotation exists. The background stays in the
  loss, so the network still learns that there are no cristae outside of mitochondria. The loss mask
  is carried in the second half of the target channels, see
  `synapse_net.training.loss.MaskedDiceLossPerSample`.
- The loss is optionally **weighted towards the mitochondria membrane**, which markedly improves the
  detection of cristae junctions, see `synapse_net.training.transform.membrane_proximity_weight`.

The training data is expected as hdf5 files, one per tomogram, that contain the tomogram and the
mitochondria state under the key 'raw_mitos_combined' and the cristae annotations under
'labels/cristae'. The files may be spread over several data roots and their sub-directories.
"""

import json
import os
import random
from glob import glob
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch_em
from torch_em.data import MinInstanceSampler

from .loss import MaskedDiceLossPerSample
from .supervised_training import supervised_training
from .transform import (
    AugmentedMitoStateMaskTransform, CRISTAE_VOXEL_SIZE, MitoStateMaskTransform, standardize_channel
)


def _normalize_roots(data_roots: Union[Mapping[str, str], Sequence[str], str]) -> Dict[str, str]:
    if isinstance(data_roots, str):
        data_roots = [data_roots]
    if not isinstance(data_roots, Mapping):
        data_roots = {os.path.basename(root.rstrip("/")): root for root in data_roots}
    return {name: root.rstrip("/") for name, root in data_roots.items()}


def _resolve_split(split_file, roots, keys):
    """Resolve the '<root name>/<relative path>' entries of a split file to filepaths."""
    with open(split_file) as f:
        split = json.load(f)
    # A split file may carry the data roots it was created with, so that it is self-contained.
    # Roots passed by the caller take precedence, so that the data can be moved.
    roots = {**_normalize_roots(split.get("roots", {})), **roots}

    resolved = []
    for key in keys:
        paths = []
        for entry in split.get(key, []):
            name, _, relative_path = entry.partition("/")
            if name not in roots:
                raise ValueError(f"The split file {split_file} refers to the unknown data root '{name}'.")
            paths.append(os.path.join(roots[name], relative_path))
        resolved.append(paths)
    return resolved


def get_cristae_test_paths(
    split_file: str,
    data_roots: Optional[Union[Mapping[str, str], Sequence[str], str]] = None,
) -> List[str]:
    """Find the tomograms that a split file holds out for testing.

    The training functions only use the 'train' and 'val' entries of a split file. Use this to get
    the 'test' entries, either to evaluate on them or to keep them out of a random split, which
    would otherwise train on them.

    Args:
        split_file: Path to the json file with the split.
        data_roots: The folders that contain the data. Can be left out if the split file carries
            the roots it was created with.

    Returns:
        The filepaths of the test data. Empty if the split file has no 'test' entry.

    Raises:
        ValueError: If a root referenced by `split_file` is not given.
    """
    roots = {} if data_roots is None else _normalize_roots(data_roots)
    test_paths, = _resolve_split(split_file, roots, ("test",))
    return test_paths


def get_cristae_paths(
    data_roots: Optional[Union[Mapping[str, str], Sequence[str], str]] = None,
    file_pattern: str = "*_combined.h5",
    val_fraction: float = 0.1,
    split_file: Optional[str] = None,
    exclude: Sequence[str] = (),
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Find the tomograms for cristae training and split them into training and validation data.

    The files are found recursively in each of the `data_roots` and are then shuffled with a fixed
    seed, so that the split is deterministic for a given set of files. Pass `split_file` to use an
    explicit split instead; this is the only way to reproduce the split of a previous training run,
    because the seeded split depends on which files are present.

    Args:
        data_roots: The folders that contain the training data. Either a mapping of a short name to
            the folder, or a list of folders, in which case the names are derived from them. The names
            are used to refer to the files in `split_file`. It can be left out if `split_file` contains
            the roots it was created with, and takes precedence over them, so that data can be moved.
        file_pattern: The pattern for selecting the files with training data.
        val_fraction: The fraction of the data to use for validation.
        split_file: Path to a json file with the keys 'train' and 'val', which each hold a list of
            entries of the form '<root name>/<path relative to that root>'. If given, `val_fraction`,
            `exclude` and `seed` are not used.
        exclude: Substrings of filepaths that should be excluded from the training data, for example
            to leave out volumes with unreliable annotations.
        seed: The seed for shuffling the files before splitting them.

    Returns:
        The filepaths for training.
        The filepaths for validation.

    Raises:
        ValueError: If no data roots are given, if no files are found, if a root referenced by
            `split_file` is not given, or if a file listed in `split_file` does not exist.
    """
    roots = {} if data_roots is None else _normalize_roots(data_roots)

    if split_file is not None:
        train_paths, val_paths = _resolve_split(split_file, roots, ("train", "val"))
        missing = [path for path in train_paths + val_paths if not os.path.exists(path)]
        if missing:
            raise ValueError(f"{len(missing)} files from the split file {split_file} do not exist, e.g. {missing[0]}.")
        return train_paths, val_paths

    if not roots:
        raise ValueError("No data roots were given, and the split file does not contain any.")

    paths = []
    for root in roots.values():
        paths.extend(glob(os.path.join(root, "**", file_pattern), recursive=True))
    paths = sorted(path for path in paths if not any(pattern in path for pattern in exclude))
    if len(paths) == 0:
        raise ValueError(f"Did not find any files matching {file_pattern} in {list(roots.values())}.")

    # We use a separate random state so that the global one is not affected by this function.
    random.Random(seed).shuffle(paths)

    n_val = int(len(paths) * val_fraction)
    # Make sure that we validate on at least one tomogram if there is more than one.
    if n_val == 0 and len(paths) > 1:
        n_val = 1
    n_train = len(paths) - n_val
    return paths[:n_train], paths[n_train:]


def cristae_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    raw_key: str = "raw_mitos_combined",
    label_key: str = "labels/cristae",
    patch_shape: Tuple[int, int, int] = (32, 256, 256),
    batch_size: int = 24,
    lr: float = 1.7e-4,
    n_iterations: int = int(1e5),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    initial_features: int = 32,
    early_stopping: Optional[int] = 25,
    log_image_interval: int = 50,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    overwrite: bool = False,
    seed: Optional[int] = None,
    deterministic: bool = False,
    sampler: Optional[Union[callable, bool]] = None,
    raw_transform: Optional[callable] = None,
    state_channel: int = 1,
    ignore_state_value: float = 2.0,
    membrane_w_pos: float = 3.0,
    membrane_w_neg: float = 2.0,
    membrane_band_nm: float = 12.0,
    membrane_offset_nm: float = 0.0,
    voxel_size: Tuple[float, float, float] = CRISTAE_VOXEL_SIZE,
    augmentations: bool = True,
    num_workers: int = 8,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    shuffle: bool = False,
    mixed_precision: bool = True,
    check: bool = False,
    **kwargs,
) -> None:
    """Run supervised training for cristae segmentation in electron tomograms.

    The default arguments reproduce SynapseNet's `cristae5` model: an anisotropic U-Net **without
    normalization layers** that takes the tomogram and the mitochondria state as input, predicts
    foreground and boundaries, and is trained with a per-sample masked dice loss that is weighted
    towards the mitochondria membrane.

    Two things about this recipe are easy to get wrong:

    - The loss mask is carried in the **target**, not in the model input. It is produced by the joint
      transform and consumed by `MaskedDiceLossPerSample`, so a custom `loss_fn` has to understand
      that layout.
    - Setting `augmentations=False` means training **without any augmentation at all**, because the
      joint transform occupies the slot of the dataloader that otherwise holds them.

    The normalization is part of the model: only the tomogram channel is standardized, and the
    mitochondria state channel is left untouched so that its values stay comparable against
    `ignore_state_value`.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the hdf5 files for the validation data.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the tomogram and the mitochondria state inside of the hdf5.
        label_key: The key that holds the cristae annotations inside of the hdf5.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The maximal number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch shape and the size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch shape and the size of the volumes used for validation.
        initial_features: The number of features in the first level of the U-Net.
            This has no effect if the model is initialized from `checkpoint_path`.
        early_stopping: The number of epochs without improvement after which training is stopped.
        log_image_interval: The interval (in iterations) at which images are written to the training log.
        checkpoint_path: Path to a model checkpoint to initialize the weights from. Only the
            weights are loaded; pass `resume` instead to continue a previous run.
        resume: Whether to continue the previous run with this name in `save_root`, restoring
            its weights, optimizer and iteration count. `n_iterations` then counts the total
            iterations, including the ones the previous run already did.
        overwrite: Whether to replace the checkpoints of a previous run with this name.
        seed: The seed for the random number generators used in training. By default it is
            not set, so that repeated runs differ.
        deterministic: Whether to also disable cudnn benchmarking when a `seed` is given.
        sampler: Sampler to accept or reject patches for training.
            By default a minimum instance sampler with a rejection probability of 0.95 is used.
        raw_transform: Transformation applied to the input before it is passed to the network.
            By default the tomogram channel is standardized and the state channel is left unchanged.
        state_channel: The channel of `raw_key` that holds the mitochondria state.
        ignore_state_value: The state value that is excluded from the loss, i.e. the value marking
            mitochondria without cristae annotations.
        membrane_w_pos: The loss weight for cristae voxels close to the mitochondria membrane.
        membrane_w_neg: The loss weight for non-cristae voxels close to the mitochondria membrane.
            Pass 1.0 for both weights to train without membrane weighting.
        membrane_band_nm: The thickness of the membrane band in nanometer.
        membrane_offset_nm: Start the membrane band this far inside the membrane.
        voxel_size: The voxel size of the training data in nanometer, in the order (z, y, x).
            It is only used to convert the membrane band from nanometer to voxels.
        augmentations: Whether to augment the training data. Note that this is not a no-op switch:
            with `False` the training runs without any augmentation.
        num_workers: The number of workers for the dataloaders.
        persistent_workers: Whether to keep the dataloader workers alive between epochs.
        prefetch_factor: The number of batches prefetched by each dataloader worker.
        shuffle: Whether to shuffle the datasets in the dataloaders. This is disabled by default,
            to match the published model; enabling it is recommended for new trainings, as it mixes
            the tomograms within a batch.
        mixed_precision: Whether to train with mixed precision.
        check: Whether to check the training and validation loaders instead of running training.
        kwargs: Additional keyword arguments for `synapse_net.training.supervised_training`.
    """
    if sampler is None:
        sampler = MinInstanceSampler(p_reject=0.95)
    if raw_transform is None:
        raw_transform = standardize_channel

    mask_kwargs = dict(
        mito_channel=state_channel, exclude_state_value=ignore_state_value,
        band_nm=membrane_band_nm, offset_nm=membrane_offset_nm,
        w_pos=membrane_w_pos, w_neg=membrane_w_neg, voxel_size=voxel_size,
    )
    if augmentations:
        transform = AugmentedMitoStateMaskTransform(torch_em.transform.get_augmentations(3), **mask_kwargs)
    else:
        transform = MitoStateMaskTransform(**mask_kwargs)

    supervised_training(
        name=name,
        train_paths=tuple(train_paths),
        val_paths=tuple(val_paths),
        label_key=label_key,
        patch_shape=patch_shape,
        save_root=save_root,
        raw_key=raw_key,
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
        sampler=sampler,
        n_samples_train=n_samples_train,
        n_samples_val=n_samples_val,
        check=check,
        loss_fn=MaskedDiceLossPerSample(),
        in_channels=2,
        out_channels=2,
        initial_features=initial_features,
        norm=None,
        transform=transform,
        checkpoint_path=checkpoint_path,
        resume=resume,
        overwrite=overwrite,
        seed=seed,
        deterministic=deterministic,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        log_image_interval=log_image_interval,
        raw_transform=raw_transform,
        with_channels=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        shuffle=shuffle,
        **kwargs,
    )


def main():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for cristae segmentation in electron tomograms.\n\n"
        "The training data has to be stored in hdf5 files, one per tomogram, which contain the tomogram and the\n"
        "mitochondria state in one dataset of shape (2, z, y, x), and the cristae annotations in another. The files\n"
        "are found recursively in the folders passed via '-i'. For example:\n"
        "synapse_net.run_cristae_training -n my_cristae_model -i /path/to/tomograms\n"
        "The trained model will be saved in the folder 'checkpoints/my_cristae_model'.\n"
        "The default arguments reproduce the training of SynapseNet's 'cristae5' model. Note that this model was\n"
        "trained with a batch size of 24 and a patch shape of 32 x 256 x 256, which requires a large GPU.\n"
        "Check out the information below for details on the arguments of this function.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-n", "--name", required=True, help="The name of the model to be trained.")
    parser.add_argument("-i", "--data_root", nargs="+", help="One or more folders with the training data. They are searched recursively. Can be left out if 'split_file' contains the data roots.")  # noqa
    parser.add_argument("--file_pattern", default="*_combined.h5", help="The pattern for selecting the files with training data.")  # noqa
    parser.add_argument("--raw_key", default="raw_mitos_combined",
                        help="The internal path for the tomogram and the mitochondria state.")
    parser.add_argument("--label_key", default="labels/cristae", help="The internal path for the cristae annotations.")  # noqa

    # How to split the data into training and validation data.
    parser.add_argument("--split_file", help="A json file with the keys 'train' and 'val', which each hold a list of entries '<root name>/<relative path>'. If not given, the data is split randomly.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.1, help="The fraction of the data to use for validation. This has no effect if 'split_file' was passed.")  # noqa
    parser.add_argument("--exclude", nargs="*", default=[], help="Substrings of filepaths to exclude from the training data.")  # noqa
    parser.add_argument("--exclude_test_from", help="A split file whose 'test' entries are excluded from the training data. Use this when splitting randomly, so that the run does not train on the test volumes of a previous split.")  # noqa
    parser.add_argument("--seed", type=int, default=42, help="The seed for shuffling the files before splitting them, and for the random number generators used in training.")  # noqa

    # The training hyperparameters.
    parser.add_argument("-p", "--patch_shape", nargs=3, type=int, default=[32, 256, 256],
                        help="The patch shape for training in ZYX.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1.7e-4, help="The initial learning rate.")
    parser.add_argument("--n_iterations", type=int, default=int(1e5), help="The maximal number of iterations to train for.")  # noqa
    parser.add_argument("--n_samples_train", type=int, help="The number of samples per epoch for training. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--n_samples_val", type=int, help="The number of samples per epoch for validation. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--initial_features", type=int, default=32, help="The number of features in the first level of the U-Net.")  # noqa
    parser.add_argument("--early_stopping", type=int, default=25, help="The number of epochs without improvement after which training is stopped.")  # noqa
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the dataloaders.")
    parser.add_argument("--no_augmentations", action="store_true", help="Train without any augmentation. This is not recommended, it is only here to reproduce earlier trainings.")  # noqa
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the datasets in the dataloaders. This is recommended for new trainings, but was not used for the published model.")  # noqa

    # The loss masking and the membrane weighting.
    parser.add_argument("--state_channel", type=int, default=1, help="The channel holding the mitochondria state.")
    parser.add_argument("--ignore_state_value", type=float, default=2.0, help="The mitochondria state that is excluded from the loss, i.e. the value marking mitochondria without cristae annotations.")  # noqa
    parser.add_argument("--membrane_w_pos", type=float, default=3.0, help="The loss weight for cristae voxels close to the mitochondria membrane.")  # noqa
    parser.add_argument("--membrane_w_neg", type=float, default=2.0, help="The loss weight for non-cristae voxels close to the mitochondria membrane. Pass 1.0 for both weights to train without membrane weighting.")  # noqa
    parser.add_argument("--membrane_band_nm", type=float, default=12.0, help="The thickness of the membrane band in nanometer.")  # noqa
    parser.add_argument("--membrane_offset_nm", type=float, default=0.0, help="Start the membrane band this far inside the membrane.")  # noqa
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.74, 1.74, 1.74], help="The voxel size of the training data in nanometer, in ZYX. Only used for the membrane band.")  # noqa

    # Where to save the model, and how to initialize it.
    parser.add_argument("--save_root", help="Root path for saving the checkpoint and log dir.")
    parser.add_argument("--checkpoint_path", help="A model checkpoint to initialize the weights from.")
    parser.add_argument("--resume", action="store_true", help="Continue the previous run with the same name in 'save_root', restoring its weights, optimizer and iteration count. '--n_iterations' is the total number of iterations, including the ones already done.")  # noqa
    parser.add_argument("--overwrite", action="store_true", help="Replace the checkpoints of a previous run with the same name. By default training refuses to overwrite them.")  # noqa
    parser.add_argument("--deterministic", action="store_true", help="Disable cudnn benchmarking so that runs with the same seed match exactly. This costs throughput.")  # noqa
    parser.add_argument("--check", action="store_true", help="Visualize samples from the data loaders to ensure correct data instead of running training.")  # noqa
    args = parser.parse_args()

    exclude = list(args.exclude)
    if args.exclude_test_from is not None:
        exclude += get_cristae_test_paths(args.exclude_test_from, args.data_root)

    train_paths, val_paths = get_cristae_paths(
        args.data_root, file_pattern=args.file_pattern, val_fraction=args.val_fraction,
        split_file=args.split_file, exclude=exclude, seed=args.seed,
    )
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths), "tomograms.")

    checkpoint_path = args.checkpoint_path

    cristae_training(
        name=args.name, train_paths=train_paths, val_paths=val_paths, save_root=args.save_root,
        raw_key=args.raw_key, label_key=args.label_key, patch_shape=tuple(args.patch_shape),
        batch_size=args.batch_size, lr=args.learning_rate, n_iterations=args.n_iterations,
        n_samples_train=args.n_samples_train, n_samples_val=args.n_samples_val,
        initial_features=args.initial_features, early_stopping=args.early_stopping,
        checkpoint_path=checkpoint_path, resume=args.resume, overwrite=args.overwrite,
        seed=args.seed, deterministic=args.deterministic,
        num_workers=args.num_workers, shuffle=args.shuffle,
        state_channel=args.state_channel, ignore_state_value=args.ignore_state_value,
        membrane_w_pos=args.membrane_w_pos, membrane_w_neg=args.membrane_w_neg,
        membrane_band_nm=args.membrane_band_nm, membrane_offset_nm=args.membrane_offset_nm,
        voxel_size=tuple(args.voxel_size), augmentations=not args.no_augmentations, check=args.check,
    )
