import os
from typing import Optional, Tuple

import torch
import torch_em.self_training as self_training

from .dataloaders import get_supervised_loader, get_unsupervised_loader
from .domain_adaptation import mean_teacher_adaptation
from .models import get_2d_model, get_3d_model
from .supervised_training import supervised_training


def semisupervised_training(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    supervised_train_paths: Tuple[str],
    supervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    label_key: str,
    save_root: str,
    raw_key: str = "raw",
    confidence_threshold: float = 0.9,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    teacher_warmup_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    source_checkpoint=None,
    supervised_sampler: Optional[callable] = None,
    check: bool = False,
):
    """Run semisupervised segmentation training.

    This proceeds in two phases:

        1. If no `source_checkpoint` is given, run supervised training to
            warmup the teacher.
        2. Run semisupervised training using mean teacher setup with invertible 
            augumentation, using the warmup checkpoint to initialize the teacher.

    Args:
        name: The name for the checkpoint to be trained. The warmup checkpoint is saved
            under the name "{name}-warmup".
        unsupervised_train_paths: Filepaths to the hdf5 files for the unsupervised
            training data. This data does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 files for the unsupervised
            validation data. This data does not require labels.
        supervised_train_paths: Filepaths to the hdf5 files for the supervised training
            data, requires labels. Used for both the teacher warmup and the semi-supervised loss.
        supervised_val_paths: Filepaths to the hdf5 files for the supervised validation data,
            requires labels.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        label_key: The key that holds the labels inside of the hdf5 files.
        save_root: Folder where the checkpoints will be saved.
        raw_key: The key that holds the raw data inside of the hdf5 files.
        confidence_threshold: The threshold for filtering data in the unsupervised loss.
            The label filtering is done based on the uncertainty of network predictions, and only
            the data with higher certainty than this threshold is used for training.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of mean-teacher training iterations.
        teacher_warmup_iterations: The number of iterations for the supervised teacher warmup,
            only used if no warmup checkpoint exists yet and no `source_checkpoint` is given.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        source_checkpoint: Warmup checkpoint used to initialize the teacher model. If not provided,
            run supervised training `teacher_warmup_iterations`.
        supervised_sampler: Optional sampler for selecting patches from the labelled data.
        check: Whether to check the training and validation loaders instead of running training.
    """
    # check both sets of loaders before teacher warmup
    if check:
        from torch_em.util.debug import check_loader

        unsupervised_train_loader = get_unsupervised_loader(
            unsupervised_train_paths, raw_key,
            patch_shape, batch_size, n_samples_train,
        )
        unsupervised_val_loader = get_unsupervised_loader(
            unsupervised_val_paths, raw_key, 
            patch_shape, batch_size, n_samples_val,
        )
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths, raw_key, label_key,
            patch_shape, batch_size, n_samples_train,
            sampler=supervised_sampler,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths, raw_key, label_key,
            patch_shape, batch_size, n_samples_val,
            sampler=supervised_sampler,
        )
        check_loader(unsupervised_train_loader, n_samples=2)
        check_loader(unsupervised_val_loader, n_samples=2)
        check_loader(supervised_train_loader, n_samples=2)
        check_loader(supervised_val_loader, n_samples=2)
        
        return

    warmup_name = f"{name}-warmup"
    if source_checkpoint is None:
        warmup_checkpoint = os.path.join(save_root, "checkpoints", warmup_name, "best.pt")

        if not os.path.exists(warmup_checkpoint):
            print(f"No warmup checkpoint was found, initiating supervised warmup for teacher model with {teacher_warmup_iterations} iterations.")

            supervised_training(
                name=warmup_name,
                train_paths=supervised_train_paths,
                val_paths=supervised_val_paths,
                label_key=label_key,
                patch_shape=patch_shape,
                save_root=save_root,
                batch_size=batch_size,
                lr=lr,
                n_iterations=teacher_warmup_iterations,
                sampler=supervised_sampler,
                check=False,
            )
        source_checkpoint = os.path.dirname(warmup_checkpoint)
    
    mean_teacher_adaptation(
        name=name,
        unsupervised_train_paths=unsupervised_train_paths,
        unsupervised_val_paths=unsupervised_val_paths,
        supervised_train_paths=supervised_train_paths,
        supervised_val_paths=supervised_val_paths,
        raw_key=raw_key,
        raw_key_supervised=raw_key,
        label_key=label_key,
        patch_shape=patch_shape,
        save_root=save_root,
        source_checkpoint=source_checkpoint,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
        n_samples_train=n_samples_train,
        n_samples_val=n_samples_val,
        supervised_sampler=supervised_sampler,
        check=False,
    )

def semisupervised_training_v0( #TODO remove old unused version?
    name: str,
    train_paths: Tuple[str],
    val_paths: Tuple[str],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: str,
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    check: bool = False,
) -> None:
    """Run semi-supervised segmentation training.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the df5 files for the validation data.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
    """
    train_loader = get_supervised_loader(train_paths, raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val)

    unsupervised_train_loader = get_unsupervised_loader(train_paths, raw_key, patch_shape, batch_size,
                                                        n_samples=n_samples_train)
    unsupervised_val_loader = get_unsupervised_loader(val_paths, raw_key, patch_shape, batch_size,
                                                      n_samples=n_samples_val)

    # TODO check the semisup loader
    if check:
        # from torch_em.util.debug import check_loader
        # check_loader(train_loader, n_samples=4)
        # check_loader(val_loader, n_samples=4)
        return

    # Check for 2D or 3D training
    is_2d = False
    z, y, x = patch_shape
    is_2d = z == 1

    if is_2d:
        model = get_2d_model(out_channels=2)
    else:
        model = get_3d_model(out_channels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Self training functionality.
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=0.9)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
    )
    trainer.fit(n_iterations)
