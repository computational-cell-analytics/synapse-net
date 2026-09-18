"""Training for mitochondria segmentation in electron tomography.

This module reproduces the recipe that was used to train SynapseNet's `mitochondria2` model,
which segments mitochondria in electron tomograms at a resolution of 2.87 nm.
It is a thin layer on top of `synapse_net.training.supervised_training` that fixes the
hyperparameters and the data layout of that model.

The training data is expected as hdf5 files, one per tomogram, that contain the tomogram
under the key 'raw' and the corresponding instance annotations under 'labels/mitochondria'.
The files may be spread over sub-directories of the data root; they are found recursively.
"""

import json
import os
import random
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import torch_em
from torch_em.data import MinInstanceSampler

from .supervised_training import _resolve_resume_checkpoint, supervised_training


def get_mitochondria_paths(
    data_root: str,
    file_pattern: str = "*.h5",
    val_fraction: float = 0.15,
    split_file: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Find the tomograms for mitochondria training and split them into training and validation data.

    The files are found recursively in `data_root` and are then shuffled with a fixed seed,
    so that the split is deterministic for a given set of files. Pass `split_file` to use an
    explicit split instead, for example to reproduce the split of a previous training run.

    Args:
        data_root: The folder that contains the training data.
        file_pattern: The pattern for selecting the files with training data.
        val_fraction: The fraction of the data to use for validation.
        split_file: Path to a json file with the keys 'train' and 'val', which each hold a list of
            filepaths relative to `data_root`. If given, `val_fraction` and `seed` are not used.
        seed: The seed for shuffling the files before splitting them.

    Returns:
        The filepaths for training.
        The filepaths for validation.

    Raises:
        ValueError: If no files are found, or if a file listed in `split_file` does not exist.
    """
    if split_file is not None:
        with open(split_file) as f:
            split = json.load(f)
        train_paths = [os.path.join(data_root, name) for name in split["train"]]
        val_paths = [os.path.join(data_root, name) for name in split["val"]]
        missing = [path for path in train_paths + val_paths if not os.path.exists(path)]
        if missing:
            raise ValueError(f"{len(missing)} files from the split file {split_file} do not exist, e.g. {missing[0]}.")
        return train_paths, val_paths

    paths = sorted(glob(os.path.join(data_root, "**", file_pattern), recursive=True))
    if len(paths) == 0:
        raise ValueError(f"Did not find any files matching {file_pattern} in {data_root}.")

    # We use a separate random state so that the global one is not affected by this function.
    random.Random(seed).shuffle(paths)

    n_val = int(len(paths) * val_fraction)
    # Make sure that we validate on at least one tomogram if there is more than one.
    if n_val == 0 and len(paths) > 1:
        n_val = 1
    n_train = len(paths) - n_val
    return paths[:n_train], paths[n_train:]


def mitochondria_training(
    name: str,
    train_paths: Sequence[str],
    val_paths: Sequence[str],
    save_root: Optional[str] = None,
    raw_key: str = "raw",
    label_key: str = "labels/mitochondria",
    patch_shape: Tuple[int, int, int] = (32, 256, 256),
    batch_size: int = 8,
    lr: float = 1e-4,
    n_iterations: int = int(1.5e5),
    n_samples_train: int = 500,
    n_samples_val: int = 500,
    initial_features: int = 32,
    early_stopping: Optional[int] = 20,
    log_image_interval: int = 50,
    checkpoint_path: Optional[str] = None,
    sampler: Optional[Union[callable, bool]] = None,
    raw_transform: Optional[callable] = None,
    num_workers: int = 8,
    shuffle: bool = False,
    mixed_precision: bool = False,
    check: bool = False,
    **kwargs,
) -> None:
    """Run supervised training for mitochondria segmentation in electron tomograms.

    The default arguments reproduce SynapseNet's `mitochondria2` model: an anisotropic U-Net
    with instance normalization that predicts foreground and boundaries, trained with a dice loss
    on percentile-normalized tomograms.

    The raw transform is part of the model, not just of the training: the same normalization has to
    be applied at inference time, by passing `preprocess=torch_em.transform.raw.normalize_percentile`
    to `synapse_net.inference.mitochondria.segment_mitochondria`. A model trained with one
    normalization and applied with another will not predict anything useful.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the hdf5 files for the validation data.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the tomogram inside of the hdf5.
        label_key: The key that holds the mitochondria annotations inside of the hdf5.
        patch_shape: The patch shape used for a training example.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The maximal number of iterations to train for.
        n_samples_train: The number of train samples per epoch.
        n_samples_val: The number of val samples per epoch.
        initial_features: The number of features in the first level of the U-Net.
            This has no effect if the model is initialized from `checkpoint_path`.
        early_stopping: The number of epochs without improvement after which training is stopped.
        log_image_interval: The interval (in iterations) at which images are written to the training log.
        checkpoint_path: Path to a model checkpoint to initialize the weights from.
        sampler: Sampler to accept or reject patches for training.
            By default a minimum instance sampler with a rejection probability of 0.95 is used.
        raw_transform: Transformation applied to the tomogram before it is passed to the network.
            By default the tomogram is normalized with the 1st and 99th percentile.
        num_workers: The number of workers for the dataloaders.
        shuffle: Whether to shuffle the datasets in the dataloaders. This is disabled by default,
            to match the published model; enabling it is recommended for new trainings, as it mixes
            the tomograms within a batch.
        mixed_precision: Whether to train with mixed precision. This is disabled by default, to match
            the published model; enabling it reduces the memory consumption and speeds up training.
        check: Whether to check the training and validation loaders instead of running training.
        kwargs: Additional keyword arguments for `synapse_net.training.supervised_training`.
    """
    if sampler is None:
        sampler = MinInstanceSampler(p_reject=0.95)
    if raw_transform is None:
        raw_transform = torch_em.transform.raw.normalize_percentile

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
        checkpoint_path=checkpoint_path,
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
        description="Train a model for mitochondria segmentation in electron tomograms.\n\n"
        "The training data has to be stored in hdf5 files, one per tomogram, which contain the tomogram and the\n"
        "mitochondria annotations. The files are found recursively in the folder passed via '-i'. For example:\n"
        "synapse_net.run_mitochondria_training -n my_mito_model -i /path/to/tomograms\n"
        "The trained model will be saved in the folder 'checkpoints/my_mito_model'.\n"
        "The default arguments reproduce the training of SynapseNet's 'mitochondria2' model. Note that this model\n"
        "was trained with a batch size of 8 and a patch shape of 32 x 256 x 256, which requires a large GPU.\n"
        "Check out the information below for details on the arguments of this function.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-n", "--name", required=True, help="The name of the model to be trained.")
    parser.add_argument("-i", "--data_root", required=True,
                        help="The folder with the training data. It is searched recursively for matching files.")
    parser.add_argument("--file_pattern", default="*.h5", help="The pattern for selecting the files with training data.")  # noqa
    parser.add_argument("--raw_key", default="raw", help="The internal path for the tomogram.")
    parser.add_argument("--label_key", default="labels/mitochondria",
                        help="The internal path for the mitochondria annotations.")

    # How to split the data into training and validation data.
    parser.add_argument("--split_file", help="A json file with the keys 'train' and 'val', which each hold a list of filepaths relative to 'data_root'. If not given, the data is split randomly.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.15, help="The fraction of the data to use for validation. This has no effect if 'split_file' was passed.")  # noqa
    parser.add_argument("--seed", type=int, default=42, help="The seed for shuffling the files before splitting them.")

    # The training hyperparameters.
    parser.add_argument("-p", "--patch_shape", nargs=3, type=int, default=[32, 256, 256],
                        help="The patch shape for training in ZYX.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate.")
    parser.add_argument("--n_iterations", type=int, default=int(1.5e5), help="The maximal number of iterations to train for.")  # noqa
    parser.add_argument("--n_samples_train", type=int, default=500, help="The number of samples per epoch for training.")  # noqa
    parser.add_argument("--n_samples_val", type=int, default=500, help="The number of samples per epoch for validation.")  # noqa
    parser.add_argument("--initial_features", type=int, default=32, help="The number of features in the first level of the U-Net.")  # noqa
    parser.add_argument("--early_stopping", type=int, default=20, help="The number of epochs without improvement after which training is stopped.")  # noqa
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the dataloaders.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the datasets in the dataloaders. This is recommended for new trainings, but was not used for the published model.")  # noqa
    parser.add_argument("--mixed_precision", action="store_true", help="Train with mixed precision. This reduces the memory consumption, but was not used for the published model.")  # noqa

    # Where to save the model, and how to initialize it.
    parser.add_argument("--save_root", help="Root path for saving the checkpoint and log dir.")
    parser.add_argument("--checkpoint_path", help="A model checkpoint to initialize the weights from.")
    parser.add_argument("--resume", action="store_true", help="Initialize the model from the best checkpoint of a previous run with the same name in 'save_root'. Note that the optimizer state is not restored.")  # noqa
    parser.add_argument("--check", action="store_true", help="Visualize samples from the data loaders to ensure correct data instead of running training.")  # noqa
    args = parser.parse_args()

    train_paths, val_paths = get_mitochondria_paths(
        args.data_root, file_pattern=args.file_pattern, val_fraction=args.val_fraction,
        split_file=args.split_file, seed=args.seed,
    )
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths), "tomograms.")

    checkpoint_path = args.checkpoint_path
    if args.resume:
        checkpoint_path = _resolve_resume_checkpoint(args.save_root, args.name, checkpoint_path)

    mitochondria_training(
        name=args.name, train_paths=train_paths, val_paths=val_paths, save_root=args.save_root,
        raw_key=args.raw_key, label_key=args.label_key, patch_shape=tuple(args.patch_shape),
        batch_size=args.batch_size, lr=args.learning_rate, n_iterations=args.n_iterations,
        n_samples_train=args.n_samples_train, n_samples_val=args.n_samples_val,
        initial_features=args.initial_features, early_stopping=args.early_stopping,
        checkpoint_path=checkpoint_path, num_workers=args.num_workers, shuffle=args.shuffle,
        mixed_precision=args.mixed_precision, check=args.check,
    )
