import os
from typing import Optional, Tuple

from .domain_adaptation import mean_teacher_adaptation
from .supervised_training import supervised_training


def semisupervised_training(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    save_root: str,
    supervised_train_paths: Tuple[str],
    supervised_val_paths: Tuple[str],
    label_key: str,
    source_checkpoint=None,
    confidence_threshold: float = 0.9,
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    teacher_warmup_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    train_mask_paths: Optional[Tuple[str]] = None,
    val_mask_paths: Optional[Tuple[str]] = None,
    sample_mask_key: Optional[str] = None,
    unsupervised_sampler: Optional[callable] = None,
    supervised_sampler: Optional[callable] = None,
    backbone: Optional[str] = None,
    model_type: Optional[str] = None,
    aug_dict: Optional[dict] = None,
    separate_backward: bool = False,
    check: bool = False,
):
    """Run semisupervised segmentation training.

    This proceeds in two phases:

        1. If no `source_checkpoint` is given, run supervised training to
            warmup the teacher.
        2. Run semisupervised training using mean teacher setup with invertible
            augmentations, using the warmup checkpoint to initialize the teacher.

    Args:
        name: The name for the checkpoint to be trained. The warmup checkpoint is saved
            under the name "{name}-warmup".
        unsupervised_train_paths: Filepaths to the hdf5 files for the unsupervised
            training data. This data does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 files for the unsupervised
            validation data. This data does not require labels.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoints will be saved.
        supervised_train_paths: Filepaths to the hdf5 files for the supervised training
            data, requires labels. Used for both the teacher warmup and the semi-supervised loss.
        supervised_val_paths: Filepaths to the hdf5 files for the supervised validation data,
            requires labels.
        label_key: The key that holds the labels inside of the hdf5 files.
        source_checkpoint: Warmup checkpoint used to initialize the teacher model. If not provided,
            run supervised training for `teacher_warmup_iterations`.
        confidence_threshold: The threshold for filtering data in the unsupervised loss.
            The label filtering is done based on the uncertainty of network predictions, and only
            the data with higher certainty than this threshold is used for training.
        raw_key: The key that holds the raw data inside of the hdf5 files.
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
        train_mask_paths: Sample masks used by the unsupervised sampler to accept or reject patches for training.
        val_mask_paths: Sample masks used by the unsupervised sampler to accept or reject patches for validation.
        sample_mask_key: The key to the sample mask dataset inside each file.
        unsupervised_sampler: Sampler to accept or reject patches for the unsupervised data stream.
        supervised_sampler: Sampler to accept or reject patches for the supervised data stream.
        backbone: The pretrained ViT encoder of a UNETR model. Options: "sam", "dinov2", or "dinov3".
            Must be set together with `model_type`.
        model_type: Model type for the selected `backbone` model family, for example "vit_b" or
            "vit_b_em_organelles" for "sam". Must be set together with `backbone`.
        aug_dict: The augmentations for the teacher and the student, with keys "intensity" and "geometrical".
            Defaults to Gaussian blur and noise, plus random flips and 90-degree rotations.
        separate_backward: Whether to backpropagate each loss term separately to reduce peak memory.
        check: Whether to check the training and validation loaders instead of running training.
    """
    if check:
        mean_teacher_adaptation(
            name=name,
            unsupervised_train_paths=unsupervised_train_paths,
            unsupervised_val_paths=unsupervised_val_paths,
            patch_shape=patch_shape,
            save_root=save_root,
            source_checkpoint=source_checkpoint,
            supervised_train_paths=supervised_train_paths,
            supervised_val_paths=supervised_val_paths,
            confidence_threshold=confidence_threshold,
            raw_key=raw_key,
            raw_key_supervised=raw_key,
            label_key=label_key,
            batch_size=batch_size,
            lr=lr,
            n_iterations=n_iterations,
            n_samples_train=n_samples_train,
            n_samples_val=n_samples_val,
            train_mask_paths=train_mask_paths,
            val_mask_paths=val_mask_paths,
            sample_mask_key=sample_mask_key,
            unsupervised_sampler=unsupervised_sampler,
            supervised_sampler=supervised_sampler,
            backbone=backbone,
            model_type=model_type,
            aug_dict=aug_dict,
            separate_backward=separate_backward,
            check=True,
        )
        return

    warmup_name = f"{name}-warmup"
    if source_checkpoint is None:
        warmup_checkpoint = os.path.join(save_root, "checkpoints", warmup_name, "best.pt")

        if not os.path.exists(warmup_checkpoint):
            print(
                "No warmup checkpoint was found, initiating supervised warmup for teacher model "
                f"with {teacher_warmup_iterations} iterations."
            )

            supervised_training(
                name=warmup_name,
                train_paths=supervised_train_paths,
                val_paths=supervised_val_paths,
                label_key=label_key,
                patch_shape=patch_shape,
                save_root=save_root,
                raw_key=raw_key,
                batch_size=batch_size,
                lr=lr,
                n_iterations=teacher_warmup_iterations,
                sampler=supervised_sampler,
                n_samples_train=n_samples_train,
                n_samples_val=n_samples_val,
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
        patch_shape=patch_shape,
        save_root=save_root,
        source_checkpoint=source_checkpoint,
        supervised_train_paths=supervised_train_paths,
        supervised_val_paths=supervised_val_paths,
        confidence_threshold=confidence_threshold,
        raw_key=raw_key,
        raw_key_supervised=raw_key,
        label_key=label_key,
        batch_size=batch_size,
        lr=lr,
        n_iterations=n_iterations,
        n_samples_train=n_samples_train,
        n_samples_val=n_samples_val,
        train_mask_paths=train_mask_paths,
        val_mask_paths=val_mask_paths,
        sample_mask_key=sample_mask_key,
        unsupervised_sampler=unsupervised_sampler,
        supervised_sampler=supervised_sampler,
        backbone=backbone,
        model_type=model_type,
        aug_dict=aug_dict,
        separate_backward=separate_backward,
        check=False,
    )
