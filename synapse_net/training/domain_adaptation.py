import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, Callable
import time

import mrcfile
import torch
import torch_em
import torch_em.self_training as self_training
from torch_em.self_training.logger import SelfTrainingTensorboardLogger
from elf.io import open_file
from sklearn.model_selection import train_test_split

from .semisupervised_training import get_unsupervised_loader
from .supervised_training import (
    get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim, _derive_key_from_files
)
from ..inference.inference import get_model_path, compute_scale_from_voxel_size
from ..inference.util import _Scaler

class PseudoLabelerWithBackgroundMask(self_training.DefaultPseudoLabeler):
    """Subclass of DefaultPseudoLabeler, which can subtract background from the pseudo labels if a background mask is provided.
        By default, assumes that the first channel contains the transformed raw data and the second channel contains the background mask.

    Args:
        confidence_mask_channel: A specific channel to use for computing the confidence mask.
            By default the confidence mask is computed across all channels independently.
            This is useful, if only one of the channels encodes a probability.
        raw_channel: Channel index of the raw data, which will be used as input to the teacher model
        background_mask_channel: Channel index of the background mask, which will be subtracted from the pseudo labels.
        kwargs: Additional keyword arguments for `self_training.DefaultPseudoLabeler`.
    """
    def __init__(
        self,
        confidence_mask_channel: Optional[int] = None,
        raw_channel: Optional[int] = 0,
        background_mask_channel: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_mask_channel = confidence_mask_channel
        self.raw_channel = raw_channel
        self.background_mask_channel = background_mask_channel
        
    def _subtract_background(self, pseudo_labels: torch.Tensor, background_mask: torch.Tensor):
        bool_mask = background_mask.bool()
        return pseudo_labels.masked_fill(bool_mask, 0)

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-labels.

        Args:
            teacher: The teacher model.
            input_: The input for this batch.

        Returns:
            The pseudo-labels.
        """
        if input_.ndim != 5:
            raise ValueError(f"Expect data with 5 dimensions (B, C, D, H, W), got shape {input_.shape}.")
        
        has_background_mask = input_.shape[1] > 1 

        if has_background_mask:
            if self.background_mask_channel > input_.shape[1]:
                raise ValueError(f"Channel index {self.background_mask_channel} is out of bounds for shape {input_.shape}.")

            background_mask = input_[:, self.background_mask_channel].unsqueeze(1)
            input_ = input_[:, self.raw_channel].unsqueeze(1)

        pseudo_labels = teacher(input_)

        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
        if self.confidence_threshold is None:
            label_mask = None
        else:
            mask_input = pseudo_labels if self.confidence_mask_channel is None\
                else pseudo_labels[self.confidence_mask_channel:(self.confidence_mask_channel+1)]
            label_mask = self._compute_label_mask_both_sides(mask_input) if self.threshold_from_both_sides\
                else self._compute_label_mask_one_side(mask_input)
            if self.confidence_mask_channel is not None:
                size = (pseudo_labels.shape[0], pseudo_labels.shape[1], *([-1] * (pseudo_labels.ndim - 2)))
                label_mask = label_mask.expand(*size)
        
        if has_background_mask:
            pseudo_labels = self._subtract_background(pseudo_labels, background_mask)

        return pseudo_labels, label_mask

class MeanTeacherTrainerWithBackgroundMask(self_training.MeanTeacherTrainer):
    """Subclass of MeanTeacherTrainer, updated to handle cases where the background mask is provided. 
    Once the pseudo labels are computed, the second channel of the teacher input is dropped, if it exists.
    The second channel of the student input is also dropped, if it exists, since it is not needed for training.

    Args:
        kwargs: Additional keyword arguments for `self_training.MeanTeacherTrainer`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_epoch_unsupervised(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        # Sample from both the supervised and unsupervised loader.
        for xu1, xu2 in self.unsupervised_train_loader:
            
            # Keep only the first channel for xu2 (student input).
            if xu2.ndim != 5:
                raise ValueError(f"Expect xu2 to have 5 dimensions (B, C, D, H, W), got shape {xu2.shape}.")
            if xu2.shape[1] > 1:
                xu2 = xu2[:, :1].contiguous()

            xu1, xu2 = xu1.to(self.device, non_blocking=True), xu2.to(self.device, non_blocking=True)

            teacher_input, model_input = xu1, xu2
            
            with forward_context(), torch.no_grad():
                # Compute the pseudo labels.
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, teacher_input)

            # Drop the second channel for xu1 (teacher input) after computing the pseudo labels.
            if xu1.ndim != 5:
                raise ValueError(f"Expect xu1 to have 5 dimensions (B, C, D, H, W), got shape {xu1.shape}.")
            if xu1.shape[1] > 1:
                xu1 = xu1[:, :1].contiguous()

            # If we have a sampler then check if the current batch matches the condition for inclusion in training.
            if self.sampler is not None:
                keep_batch = self.sampler(pseudo_labels, label_filter)
                if not keep_batch:
                    continue

            self.optimizer.zero_grad()
            # Perform unsupervised training
            with forward_context():
                loss = self.unsupervised_loss(self.model, model_input, pseudo_labels, label_filter)
            backprop(loss)

            if self.logger is not None:
                with torch.no_grad(), forward_context():
                    pred = self.model(model_input) if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train_unsupervised(
                    self._iteration, loss, xu1, xu2, pred, pseudo_labels, label_filter
                )
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_lr(self._iteration, lr)
                if self.pseudo_labeler.confidence_threshold is not None:
                    self.logger.log_ct(self._iteration, self.pseudo_labeler.confidence_threshold)

            with torch.no_grad():
                self._momentum_update()

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

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
    train_sample_mask_paths: Optional[Tuple[str]] = None,
    val_sample_mask_paths: Optional[Tuple[str]] = None,
    train_background_mask_paths: Optional[Tuple[str]] = None,
    patch_sampler: Optional[callable] = None,
    pseudo_label_sampler: Optional[callable] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Run domain adapation to transfer a network trained on a source domain for a supervised
    segmentation task to perform this task on a different target domain.

    We support different domain adaptation settings:
    - unsupervised domain adaptation: the default mode when 'supervised_train_paths' and
     'supervised_val_paths' are not given.
    - semi-supervised domain adaptation: domain adaptation on unlabeled and labeled data,
      when 'supervised_train_paths' and 'supervised_val_paths' are given.

    Args:
        name: The name for the checkpoint to be trained.
        unsupervsied_train_paths: Filepaths to the hdf5 or mrc files for the training data in the target domain.
            This training data is used for unsupervised learning, so it does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 or mrc files for the validation data in the target domain.
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
        train_sample_mask_paths: Filepaths to the sample masks used by the patch sampler to accept or reject 
            patches for training.
        val_sample_mask_paths: Filepaths to the sample masks mrc files used by the patch sampler to accept or reject 
            patches for validation. 
        train_background_mask_paths: Filepaths to the background masks mrc files used for training.
            Background masks are used to subtract background from the pseudo labels before the forward pass. 
        patch_sampler: A sampler for rejecting patches based on a defined conditon. 
        pseudo_label_sampler: A sampler for rejecting pseudo-labels based on a defined condition.
        device: GPU ID for training. 
    """
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
    if train_background_mask_paths is not None:
        pseudo_labeler = PseudoLabelerWithBackgroundMask(confidence_threshold=confidence_threshold, background_mask_channel=1)
        trainer_class = MeanTeacherTrainerWithBackgroundMask
    else:
        pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)
        trainer_class = self_training.MeanTeacherTrainer

    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()
   
    unsupervised_train_loader = get_unsupervised_loader(
        data_paths=unsupervised_train_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_train, 
        sample_mask_paths=train_sample_mask_paths, 
        background_mask_paths=train_background_mask_paths,
        sampler=patch_sampler
    )
    unsupervised_val_loader = get_unsupervised_loader(
        data_paths=unsupervised_val_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_val, 
        sample_mask_paths=val_sample_mask_paths, 
        background_mask_paths=None,
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

    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    trainer = trainer_class(
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
    parser.add_argument(
        "--source_model",
        default="vesicles_3d",
        help="The source model used for weight initialization of teacher and student model. "
        "By default the model 'vesicles_3d' for vesicle segmentation in volumetric data is used."
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

    args = parser.parse_args()

    source_checkpoint = get_model_path(args.source_model)
    patch_shape = _parse_patch_shape(args.patch_shape, args.source_model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        unsupervised_train_paths, unsupervised_val_paths = _get_paths(
            args.input, args.pattern, args.resize_training_data, args.source_model, tmp_dir, args.val_fraction,
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
        )