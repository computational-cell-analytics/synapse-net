import os
from typing import Optional, Tuple

from .dataloaders import get_supervised_loader, get_unsupervised_loader
from .domain_adaptation import mean_teacher_adaptation
from .models import get_raw_transform
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
    train_mask_paths: Optional[Tuple[str]] = None,
    val_mask_paths: Optional[Tuple[str]] = None,
    sample_mask_key: Optional[str] = None,
    backbone: Optional[str] = None,
    model_type: Optional[str] = None,
    source_checkpoint=None,
    supervised_sampler: Optional[callable] = None,
    unsupervised_sampler: Optional[callable] = None,
    separate_backward: bool = False,
    check: bool = False,
):
    """Run semisupervised segmentation training.

    This proceeds in two phases:

        1. If no `source_checkpoint` is given, run supervised training to
            warmup the teacher.
        2. Run semisupervised training using mean teacher setup with invertible 
            augumentations, using the warmup checkpoint to initialize the teacher.

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
            Set to 0 to stop after the teacher warmup.
        teacher_warmup_iterations: The number of iterations for the supervised teacher warmup,
            only used if no warmup checkpoint exists yet and no `source_checkpoint` is given.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        backbone: The pretrained ViT encoder of a UNETR model. Options: "sam", "dinov2", or "dinov3".
            Must be set together with `model_type`.
        model_type: Model type for the selected `backbone` model family, for example "vit_b" or "vit_t".
            Must be set together with `backbone`.
        source_checkpoint: Warmup checkpoint used to initialize the teacher model. If not provided,
            run supervised training `teacher_warmup_iterations`.
        train_mask_paths: Sample masks used by the unsupervised sampler to accept or reject patches for training.
        val_mask_paths: Sample masks used by the unsupervised sampler to accept or reject patches for validation.
        sample_mask_key: The key to the sample mask dataset inside each file.
        supervised_sampler:  Sampler to accept or reject patches for the supervised data stream.
        unsupervised_sampler:  Sampler to accept or reject patches for the unsupervised data stream.
        separate_backward: Whether to backpropagate each loss term separately to reduce peak memory.
        check: Whether to check the training and validation loaders instead of running training.
    """
    raw_transform = get_raw_transform(backbone)[0]

    # check both sets of loaders before teacher warmup
    if check:
        from torch_em.util.debug import check_loader

        unsupervised_train_loader = get_unsupervised_loader(
            unsupervised_train_paths, raw_key, patch_shape, batch_size, n_samples_train,
            sample_mask_paths=train_mask_paths, sample_mask_key=sample_mask_key,
            sampler=unsupervised_sampler, raw_transform=raw_transform,
        )
        unsupervised_val_loader = get_unsupervised_loader(
            unsupervised_val_paths, raw_key, patch_shape, batch_size, n_samples_val,
            sample_mask_paths=val_mask_paths, sample_mask_key=sample_mask_key,
            sampler=unsupervised_sampler, raw_transform=raw_transform,
        )
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths, raw_key, label_key, patch_shape, batch_size, n_samples_train,
            sampler=supervised_sampler, raw_transform=raw_transform,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths, raw_key, label_key, patch_shape, batch_size, n_samples_val,
            sampler=supervised_sampler, raw_transform=raw_transform,
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
                backbone=backbone,
                model_type=model_type,
                check=False,
            )
        source_checkpoint = os.path.dirname(warmup_checkpoint)

    if n_iterations == 0:
        return

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
        train_mask_paths=train_mask_paths,
        val_mask_paths=val_mask_paths,
        sample_mask_key=sample_mask_key,
        supervised_sampler=supervised_sampler,
        unsupervised_sampler=unsupervised_sampler,
        backbone=backbone,
        model_type=model_type,
        check=False,
        separate_backward=separate_backward,
    )