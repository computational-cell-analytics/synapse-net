"""Training for mitochondria segmentation in volume electron microscopy.

This module reproduces the recipe that was used to train SynapseNet's volume EM mitochondria model on
FIB-SEM data with a voxel size of 25 nm in z and 5 nm in xy. It is a thin layer on top of
`synapse_net.training.supervised_training` that fixes the hyperparameters and the data layout of that
model. See `synapse_net.training.mitochondria` for the corresponding electron tomography model.

Volume EM training differs from the tomography training in two ways:

- The data is **strongly anisotropic** (5:1), so the U-Net downsamples only in xy for its first two
  levels instead of one, see `VOL_EM_SCALE_FACTORS`. The network is also trained **without
  normalization layers**.
- The blocks were cut out of a larger FIB-SEM volume and carry **white filler borders** where the
  cutout extends past the imaged region. They are removed before the normalization, see
  `synapse_net.training.transform.remove_white_patches`.

The training data is expected as hdf5 files, one per block, that contain the image data under the key
'raw' and the corresponding instance annotations under 'labels/mitochondria'. The files may be spread
over several data roots and their sub-directories.
"""

import os
import random
from glob import glob
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch_em
from torch_em.data import MinInstanceSampler

from .split import _normalize_roots, _resolve_split
from .supervised_training import supervised_training
from .transform import RemoveWhitePatchesAndNormalize

VOL_EM_SCALE_FACTORS = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
"""The downscaling factors of the U-Net encoder: two anisotropic levels, for the 5:1 anisotropy of the
volume EM data. After them the voxels are approximately isotropic, so the deeper levels downsample in
all three axes.
"""


def get_vol_em_mitochondria_paths(
    data_roots: Optional[Union[Mapping[str, str], Sequence[str], str]] = None,
    file_pattern: str = "*.h5",
    val_fraction: float = 0.15,
    split_file: Optional[str] = None,
    exclude: Sequence[str] = (),
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Find the blocks for volume EM mitochondria training and split them into training and validation data.

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
            to leave out blocks with unreliable annotations.
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
    # Make sure that we validate on at least one block if there is more than one.
    if n_val == 0 and len(paths) > 1:
        n_val = 1
    n_train = len(paths) - n_val
    return paths[:n_train], paths[n_train:]


def vol_em_mitochondria_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    raw_key: str = "raw",
    label_key: str = "labels/mitochondria",
    patch_shape: Tuple[int, int, int] = (32, 512, 512),
    batch_size: int = 4,
    lr: float = 1e-4,
    n_iterations: int = int(5e4),
    n_samples_train: int = 500,
    n_samples_val: int = 500,
    initial_features: int = 32,
    scale_factors: Sequence[Sequence[int]] = VOL_EM_SCALE_FACTORS,
    norm: Optional[str] = None,
    early_stopping: Optional[int] = 20,
    log_image_interval: int = 50,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    overwrite: bool = False,
    seed: Optional[int] = None,
    deterministic: bool = False,
    sampler: Optional[Union[callable, bool]] = None,
    raw_transform: Optional[callable] = None,
    fix_white_patches: bool = True,
    white_patch_min_size: int = 20,
    num_workers: int = 8,
    shuffle: bool = False,
    mixed_precision: bool = False,
    check: bool = False,
    **kwargs,
) -> None:
    """Run supervised training for mitochondria segmentation in volume electron microscopy.

    The default arguments reproduce SynapseNet's volume EM mitochondria model: an anisotropic U-Net
    **without normalization layers** and with **two anisotropic downsampling levels** that predicts
    foreground and boundaries, trained with a dice loss on percentile-normalized FIB-SEM blocks whose
    white filler borders have been removed.

    The raw transform is part of the model, not just of the training, and it has to be reproduced at
    inference time in two steps:

    - Remove the filler from the **whole volume** with
      `synapse_net.training.transform.remove_white_patches` before segmenting. It cannot be done in
      the per-block `preprocess` of `synapse_net.inference.mitochondria.segment_mitochondria`, because
      `get_prediction` standardizes a numpy input volume before it runs that, and the filler is
      identified by its literal value, which no longer exists after standardizing.
    - Pass `preprocess=torch_em.transform.raw.normalize_percentile` to `segment_mitochondria`. The
      percentile normalization is invariant under that standardization, so the network sees the same
      values it saw during training.

    Skipping the first step is not a small deviation: the filler skews the percentile normalization
    of every tile it overlaps, which degrades the prediction on the surrounding tissue. On one block
    that is 14% filler, leaving it in cost 0.04 Dice (0.79 instead of 0.84). See
    `scripts/volume_em/inference/run_mitochondria_vol_em_segmentation.py` for a script that does both.

    Two training runs of this model exist and both are named 'final'. This function reproduces the run
    with a patch shape of 32 x 512 x 512, a batch size of 4, without mixed precision and with the
    filler removal. The other run used a patch shape of 64 x 256 x 256, a batch size of 8, mixed
    precision and no filler removal; all other hyperparameters are the same. Both process the same
    number of voxels per batch, so they trade field of view in-plane against field of view in z.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the hdf5 files for the validation data.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the image data inside of the hdf5.
        label_key: The key that holds the mitochondria annotations inside of the hdf5.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The maximal number of iterations to train for.
        n_samples_train: The number of train samples per epoch.
        n_samples_val: The number of val samples per epoch.
        initial_features: The number of features in the first level of the U-Net.
            This has no effect if the model is initialized from `checkpoint_path`.
        scale_factors: The downscaling factors for each level of the U-Net encoder, in ZYX. The
            default matches the 5:1 anisotropy of the volume EM data.
            This has no effect if the model is initialized from `checkpoint_path`.
        norm: The normalization layer used in the convolutional blocks of the U-Net. The published
            model was trained without normalization layers.
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
        raw_transform: Transformation applied to the image data before it is passed to the network.
            By default the white filler borders are removed and the block is then normalized with the
            1st and 99th percentile. Passing this overrides `fix_white_patches`.
        fix_white_patches: Whether to remove the white filler borders of the blocks before normalizing
            them. Disable this for data that was not cut out of a larger volume.
        white_patch_min_size: The minimal size (in voxels) of a filler component that is removed.
        num_workers: The number of workers for the dataloaders.
        shuffle: Whether to shuffle the datasets in the dataloaders. This is disabled by default,
            to match the published model; enabling it is recommended for new trainings, as it mixes
            the blocks within a batch.
        mixed_precision: Whether to train with mixed precision. This is disabled by default, to match
            the published model; enabling it reduces the memory consumption and speeds up training.
        check: Whether to check the training and validation loaders instead of running training.
        kwargs: Additional keyword arguments for `synapse_net.training.supervised_training`.
    """
    if sampler is None:
        sampler = MinInstanceSampler(p_reject=0.95)
    if raw_transform is None:
        raw_transform = (
            RemoveWhitePatchesAndNormalize(min_size=white_patch_min_size) if fix_white_patches
            else torch_em.transform.raw.normalize_percentile
        )

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
        initial_features=initial_features,
        scale_factors=scale_factors,
        norm=norm,
        checkpoint_path=checkpoint_path,
        resume=resume,
        overwrite=overwrite,
        seed=seed,
        deterministic=deterministic,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        log_image_interval=log_image_interval,
        raw_transform=raw_transform,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs,
    )


def main():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for mitochondria segmentation in volume electron microscopy.\n\n"
        "The training data has to be stored in hdf5 files, one per block, which contain the image data and the\n"
        "mitochondria annotations. The files are found recursively in the folders passed via '-i'. For example:\n"
        "synapse_net.run_vol_em_mitochondria_training -n my_mito_model -i /path/to/blocks\n"
        "The trained model will be saved in the folder 'checkpoints/my_mito_model'.\n"
        "The default arguments reproduce the training of SynapseNet's volume EM mitochondria model, which was\n"
        "trained on FIB-SEM data with a voxel size of 25 nm in z and 5 nm in xy. Note that this model was trained\n"
        "with a batch size of 4 and a patch shape of 32 x 512 x 512, which requires a large GPU.\n"
        "Check out the information below for details on the arguments of this function.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-n", "--name", required=True, help="The name of the model to be trained.")
    parser.add_argument("-i", "--data_root", nargs="+", help="One or more folders with the training data. They are searched recursively. Can be left out if 'split_file' contains the data roots.")  # noqa
    parser.add_argument("--file_pattern", default="*.h5", help="The pattern for selecting the files with training data.")  # noqa
    parser.add_argument("--raw_key", default="raw", help="The internal path for the image data.")
    parser.add_argument("--label_key", default="labels/mitochondria",
                        help="The internal path for the mitochondria annotations.")

    # How to split the data into training and validation data.
    parser.add_argument("--split_file", help="A json file with the keys 'train' and 'val', which each hold a list of entries '<root name>/<relative path>'. If not given, the data is split randomly.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.15, help="The fraction of the data to use for validation. This has no effect if 'split_file' was passed.")  # noqa
    parser.add_argument("--exclude", nargs="*", default=[], help="Substrings of filepaths to exclude from the training data.")  # noqa
    parser.add_argument("--seed", type=int, default=42, help="The seed for shuffling the files before splitting them, and for the random number generators used in training.")  # noqa

    # The training hyperparameters.
    parser.add_argument("-p", "--patch_shape", nargs=3, type=int, default=[32, 512, 512],
                        help="The patch shape for training in ZYX.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate.")
    parser.add_argument("--n_iterations", type=int, default=int(5e4), help="The maximal number of iterations to train for.")  # noqa
    parser.add_argument("--n_samples_train", type=int, default=500, help="The number of samples per epoch for training.")  # noqa
    parser.add_argument("--n_samples_val", type=int, default=500, help="The number of samples per epoch for validation.")  # noqa
    parser.add_argument("--initial_features", type=int, default=32, help="The number of features in the first level of the U-Net.")  # noqa
    parser.add_argument("--early_stopping", type=int, default=20, help="The number of epochs without improvement after which training is stopped.")  # noqa
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the dataloaders.")
    parser.add_argument("--no_white_patch_fix", action="store_true", help="Do not remove the white filler borders of the blocks before normalizing them. Use this for data that was not cut out of a larger volume.")  # noqa
    parser.add_argument("--white_patch_min_size", type=int, default=20, help="The minimal size (in voxels) of a white filler component that is removed.")  # noqa
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the datasets in the dataloaders. This is recommended for new trainings, but was not used for the published model.")  # noqa
    parser.add_argument("--mixed_precision", action="store_true", help="Train with mixed precision. This reduces the memory consumption, but was not used for the published model.")  # noqa

    # Where to save the model, and how to initialize it.
    parser.add_argument("--save_root", help="Root path for saving the checkpoint and log dir.")
    parser.add_argument("--checkpoint_path", help="A model checkpoint to initialize the weights from.")
    parser.add_argument("--resume", action="store_true", help="Continue the previous run with the same name in 'save_root', restoring its weights, optimizer and iteration count. '--n_iterations' is the total number of iterations, including the ones already done.")  # noqa
    parser.add_argument("--overwrite", action="store_true", help="Replace the checkpoints of a previous run with the same name. By default training refuses to overwrite them.")  # noqa
    parser.add_argument("--deterministic", action="store_true", help="Disable cudnn benchmarking so that runs with the same seed match exactly. This costs throughput.")  # noqa
    parser.add_argument("--check", action="store_true", help="Visualize samples from the data loaders to ensure correct data instead of running training.")  # noqa
    args = parser.parse_args()

    train_paths, val_paths = get_vol_em_mitochondria_paths(
        args.data_root, file_pattern=args.file_pattern, val_fraction=args.val_fraction,
        split_file=args.split_file, exclude=args.exclude, seed=args.seed,
    )
    print("Training on", len(train_paths), "blocks and validating on", len(val_paths), "blocks.")

    vol_em_mitochondria_training(
        name=args.name, train_paths=train_paths, val_paths=val_paths, save_root=args.save_root,
        raw_key=args.raw_key, label_key=args.label_key, patch_shape=tuple(args.patch_shape),
        batch_size=args.batch_size, lr=args.learning_rate, n_iterations=args.n_iterations,
        n_samples_train=args.n_samples_train, n_samples_val=args.n_samples_val,
        initial_features=args.initial_features, early_stopping=args.early_stopping,
        checkpoint_path=args.checkpoint_path, resume=args.resume, overwrite=args.overwrite,
        seed=args.seed, deterministic=args.deterministic,
        fix_white_patches=not args.no_white_patch_fix, white_patch_min_size=args.white_patch_min_size,
        num_workers=args.num_workers, shuffle=args.shuffle,
        mixed_precision=args.mixed_precision, check=args.check,
    )
