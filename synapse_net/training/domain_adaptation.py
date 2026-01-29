import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import mrcfile
import torch
import torch_em
import torch_em.self_training as self_training
from elf.io import open_file
from sklearn.model_selection import train_test_split

from .semisupervised_training import get_unsupervised_loader
from .supervised_training import (
    get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim, _derive_key_from_files
)
from ..inference.inference import get_model_path, compute_scale_from_voxel_size, get_available_models
from ..inference.util import _Scaler


def mean_teacher_adaptation(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    save_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    supervised_train_paths: Optional[Tuple[str]] = None,
    supervised_val_paths: Optional[Tuple[str]] = None,
    confidence_threshold: float = 0.9,
    raw_key: str = "raw",
    raw_key_supervised: str = "raw",
    label_key: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    train_mask_paths: Optional[Tuple[str]] = None,
    val_mask_paths: Optional[Tuple[str]] = None,
    patch_sampler: Optional[callable] = None,
    pseudo_label_sampler: Optional[callable] = None,
    device: int = 0,
    check: bool = False,
) -> None:
    """Run domain adaptation to transfer a network trained on a source domain for a supervised
    segmentation task to perform this task on a different target domain.

    We support different domain adaptation settings:
    - unsupervised domain adaptation: the default mode when 'supervised_train_paths' and
     'supervised_val_paths' are not given.
    - semi-supervised domain adaptation: domain adaptation on unlabeled and labeled data,
      when 'supervised_train_paths' and 'supervised_val_paths' are given.

    Args:
        name: The name for the checkpoint to be trained.
        unsupervsied_train_paths: Filepaths to the hdf5 files or similar file formats
            for the training data in the target domain.
            This training data is used for unsupervised learning, so it does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 files or similar file formats
            for the validation data in the target domain.
            This validation data is used for unsupervised learning, so it does not require labels.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        source_checkpoint: Checkpoint to the initial model trained on the source domain.
            This is used to initialize the teacher model.
            If the checkpoint is not given, then both student and teacher model are initialized
            from scratch. In this case `supervised_train_paths` and `supervised_val_paths` have to
            be given in order to provide training data from the source domain.
        supervised_train_paths: Filepaths to the hdf5 files for the training data in the source domain.
            This training data is optional. If given, it is used for unsupervised learnig and requires labels.
        supervised_val_paths: Filepaths to the df5 files for the validation data in the source domain.
            This validation data is optional. If given, it is used for unsupervised learnig and requires labels.
        confidence_threshold: The threshold for filtering data in the unsupervised loss.
            The label filtering is done based on the uncertainty of network predictions, and only
            the data with higher certainty than this threshold is used for training.
        raw_key: The key that holds the raw data inside of the hdf5 or similar files.
        label_key: The key that holds the labels inside of the hdf5 files for supervised learning.
            This is only required if `supervised_train_paths` and `supervised_val_paths` are given.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        train_mask_paths: Sample masks used by the patch sampler to accept or reject patches for training.
        val_mask_paths: Sample masks used by the patch sampler to accept or reject patches for validation.
        patch_sampler: Accept or reject patches based on a condition.
        pseudo_label_sampler: Mask out regions of the pseudo labels where the teacher is not confident before updating the gradients.
        device: GPU ID for training.
        check: Whether to check the training and validation loaders instead of running training.
    """  # noqa
    assert (supervised_train_paths is None) == (supervised_val_paths is None)
    is_2d, _ = _determine_ndim(patch_shape)

    if source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_train_paths is not None
        print("Mean teacher training from scratch (AdaMT)")
        if is_2d:
            model = get_2d_model(out_channels=2)
        else:
            model = get_3d_model(out_channels=2)
        reinit_teacher = True
    else:
        print("Mean teacher training initialized from source model:", source_checkpoint)
        if os.path.isdir(source_checkpoint):
            model = torch_em.util.load_model(source_checkpoint)
        else:
            model = torch.load(source_checkpoint, weights_only=False)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # self training functionality
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    unsupervised_train_loader = get_unsupervised_loader(
        data_paths=unsupervised_train_paths,
        raw_key=raw_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples=n_samples_train,
        sample_mask_paths=train_mask_paths,
        sampler=patch_sampler
    )
    unsupervised_val_loader = get_unsupervised_loader(
        data_paths=unsupervised_val_paths,
        raw_key=raw_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples=n_samples_val,
        sample_mask_paths=val_mask_paths,
        sampler=patch_sampler
    )

    if supervised_train_paths is not None:
        assert label_key is not None
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, n_samples=n_samples_train,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, n_samples=n_samples_val,
        )
    else:
        supervised_train_loader = None
        supervised_val_loader = None

    if check:
        from torch_em.util.debug import check_loader
        check_loader(unsupervised_train_loader, n_samples=4)
        check_loader(unsupervised_val_loader, n_samples=4)
        if supervised_train_loader is not None:
            check_loader(supervised_train_loader, n_samples=4)
            check_loader(supervised_val_loader, n_samples=4)
        return

    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=save_root,
        sampler=pseudo_label_sampler,
    )
    trainer.fit(n_iterations)


# TODO patch shapes for other models
PATCH_SHAPES = {
    "vesicles_3d": [48, 256, 256],
}
"""@private
"""


def _get_paths(input_folder, pattern, resize_training_data, model_name, tmp_dir, val_fraction):
    files = sorted(glob(os.path.join(input_folder, "**", pattern), recursive=True))
    if len(files) == 0:
        raise ValueError(f"Could not load any files from {input_folder} with pattern {pattern}")

    # Heuristic: if we have less then 4 files then we crop a part of the volumes for validation.
    # And resave the volumes.
    resave_val_crops = len(files) < 4

    # We only resave the data if we resave val crops or resize the training data
    resave_data = resave_val_crops or resize_training_data
    if not resave_data:
        train_paths, val_paths = train_test_split(files, test_size=val_fraction)
        return train_paths, val_paths

    train_paths, val_paths = [], []
    for file_path in files:
        file_name = os.path.basename(file_path)
        data = open_file(file_path, mode="r")["data"][:]

        if resize_training_data:
            with mrcfile.open(file_path) as f:
                voxel_size = f.voxel_size
            voxel_size = {ax: vox_size / 10.0 for ax, vox_size in zip("xyz", voxel_size.item())}
            scale = compute_scale_from_voxel_size(voxel_size, model_name)
            scaler = _Scaler(scale, verbose=False)
            data = scaler.sale_input(data)

        if resave_val_crops:
            n_slices = data.shape[0]
            val_slice = int((1.0 - val_fraction) * n_slices)
            train_data, val_data = data[:val_slice], data[val_slice:]

            train_path = os.path.join(tmp_dir, Path(file_name).with_suffix(".h5")).replace(".h5", "_train.h5")
            with open_file(train_path, mode="w") as f:
                f.create_dataset("data", data=train_data, compression="lzf")
            train_paths.append(train_path)

            val_path = os.path.join(tmp_dir, Path(file_name).with_suffix(".h5")).replace(".h5", "_val.h5")
            with open_file(val_path, mode="w") as f:
                f.create_dataset("data", data=val_data, compression="lzf")
            val_paths.append(val_path)

        else:
            output_path = os.path.join(tmp_dir, Path(file_name).with_suffix(".h5"))
            with open_file(output_path, mode="w") as f:
                f.create_dataset("data", data=data, compression="lzf")
            train_paths.append(output_path)

    if not resave_val_crops:
        train_paths, val_paths = train_test_split(train_paths, test_size=val_fraction)

    return train_paths, val_paths


def _parse_patch_shape(patch_shape, model_name):
    if patch_shape is None:
        patch_shape = PATCH_SHAPES[model_name]
    return patch_shape


def main():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Adapt a model to data from a different domain using unsupervised domain adaptation.\n\n"
        "You can use this function to adapt the SynapseNet model for vesicle segmentation like this:\n"
        "synapse_net.run_domain_adaptation -n adapted_model -i /path/to/data --file_pattern *.mrc --source_model vesicles_3d\n"  # noqa
        "The trained model will be saved in the folder 'checkpoints/adapted_model' (or whichever name you pass to the '-n' argument)."  # noqa
        "You can then use this model for segmentation with the SynapseNet GUI or CLI. "
        "Check out the information below for details on the arguments of this function.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--name", "-n", required=True, help="The name of the model to be trained. ")
    parser.add_argument("--input_folder", "-i", required=True, help="The folder with the training data.")
    parser.add_argument("--file_pattern", default="*",
                        help="The pattern for selecting files for training. For example '*.mrc' to select mrc files.")
    parser.add_argument("--key", help="The internal file path for the training data. Will be derived from the file extension by default.")  # noqa
    available_models = get_available_models()
    parser.add_argument(
        "--source_model",
        default="vesicles_3d",
        help="The source model used for weight initialization of teacher and student model. "
        "By default the model 'vesicles_3d' for vesicle segmentation in volumetric data is used.\n"
        f"The following source models are available: {available_models}"
    )
    parser.add_argument(
        "--resize_training_data", action="store_true",
        help="Whether to resize the training data to fit the voxel size of the source model's trainign data."
    )
    parser.add_argument("--n_iterations", type=int, default=int(1e4), help="The number of iterations for training.")
    parser.add_argument(
        "--patch_shape", nargs=3, type=int,
        help="The patch shape for training. By default the patch shape the source model was trained with is used."
    )

    # More optional argument:
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument("--n_samples_train", type=int, help="The number of samples per epoch for training. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--n_samples_val", type=int, help="The number of samples per epoch for validation. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.15, help="The fraction of the data to use for validation. This has no effect if 'val_folder' and 'val_label_folder' were passed.")  # noqa
    parser.add_argument("--check", action="store_true", help="Visualize samples from the data loaders to ensure correct data instead of running training.")  # noqa
    parser.add_argument("--save_root", help="Root path for saving the checkpoint and log dir.")

    args = parser.parse_args()

    source_checkpoint = get_model_path(args.source_model)
    patch_shape = _parse_patch_shape(args.patch_shape, args.source_model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        unsupervised_train_paths, unsupervised_val_paths = _get_paths(
            args.input_folder, args.file_pattern, args.resize_training_data,
            args.source_model, tmp_dir, args.val_fraction,
        )
        unsupervised_train_paths, raw_key = _derive_key_from_files(unsupervised_train_paths, args.key)

        mean_teacher_adaptation(
            name=args.name,
            unsupervised_train_paths=unsupervised_train_paths,
            unsupervised_val_paths=unsupervised_val_paths,
            patch_shape=patch_shape,
            source_checkpoint=source_checkpoint,
            raw_key=raw_key,
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
            n_samples_train=args.n_samples_train,
            n_samples_val=args.n_samples_val,
            check=args.check,
            save_root=args.save_root,
        )
