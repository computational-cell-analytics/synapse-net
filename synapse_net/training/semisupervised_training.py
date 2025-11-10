from typing import Optional, Tuple, List

import numpy as np
import uuid
import h5py
import torch
import torch_em
import torch_em.self_training as self_training
from torchvision import transforms

from synapse_net.file_utils import read_mrc
from .supervised_training import get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim
from pathlib import Path

def weak_augmentations(p: float = 0.75) -> callable:
    """The weak augmentations used in the unsupervised data loader.

    Args:
        p: The probability for applying one of the augmentations.

    Returns:
        The transformation function applying the augmentation.
    """
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)

class DropChannel:
    def __init__(self, channel: int):
        self.channel = channel

    def __call__(self, data):
        if data.ndim != 4:
            raise ValueError("Expect data with 4 dimensions (C, D, H, W).")
    
        if self.channel > data.shape[0]:
            raise ValueError(f"Drop channel index {self.channel} is out of bounds for shape {data.shape}.")
        return np.delete(data, self.channel, axis=0)

class ChannelWiseRawTransform:
    def __init__(self, base_transform: callable, transform_channel: int = 0):
        self.base_transform = base_transform
        self.transform_channel = transform_channel

    def __call__(self, data):
        if data.ndim != 4:
            raise ValueError("Expect data with 4 dimensions (C, D, H, W).")

        if self.transform_channel > data.shape[0]:
            raise ValueError(f"Transform channel index {self.transform_channel} is out of bounds for shape {data.shape}.")
            
        output = data.copy()
        output[self.transform_channel] = self.base_transform(data[self.transform_channel])
        return output

class ComposedTransform:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x

# fix this to avoid serialization issues
class ChannelWiseAugmentations:
    def __init__(self, transform_channel: int = 0, base_augmentations: Optional[callable] = None):
        self.transform_channel = transform_channel
        if base_augmentations is None:
            self.base_augmentations = weak_augmentations()

    def __call__(self, data):
        if data.ndim != 4:
            raise ValueError("Expect data with 4 dimensions (C, D, H, W).") 

        if self.transform_channel > data.shape[0]:
            raise ValueError(f"Augmentations channel index {self.transform_channel} is out of bounds for shape {data.shape}.")   

        output = data.clone() 
        output[self.transform_channel] = self.base_augmentations(data[self.transform_channel])
        return output

class ChannelSplitterSampler: 
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __call__(self, x):
        raw, mask = x[0], x[1]
        return self.sampler(raw, mask)
    
def get_stacked_path(inputs: List[np.ndarray]):
    """Helper function to get_unsupervised_loader(). In cases where a mask input is given, creates a temporary 
     h5 file containing the stacked inputs (raw, sample_mask, background mask) to be used to RawDataset().

    Args:
        inputs: List of numpy arrays to be stacked along axis 0.

    Returns: Path to temporary h5 file containing stacked inputs.
    """
    stacked = np.stack(inputs, axis=0)
    tmp_path = f"/tmp/stacked_{uuid.uuid4().hex}.h5"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=stacked, compression="gzip")
    return tmp_path

def get_unsupervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int] = None,
    sample_mask_paths: Optional[Tuple[str]] = None,
    background_mask_paths: Tuple[str] = None,
    sampler: Optional[callable] = None,
    exclude_top_and_bottom: bool = False, 
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 or mrc files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        exclude_top_and_bottom: Whether to exluce the five top and bottom slices to
            avoid artifacts at the border of tomograms.
        sample_mask_paths: The mrc filepaths to the corresponding sample masks for each tomogram.
        background_mask_paths: The mrc filepaths to the corresponding background masks for each tomogram.
        sampler: Accept or reject patches based on a condition.

    Returns:
        The PyTorch dataloader.
    """
    # We exclude the top and bottom slices where the tomogram reconstruction is bad.
    if exclude_top_and_bottom:
        roi = np.s_[5:-5, :, :]
    else:
        roi = None
    
    # initialize class instances
    base_transform = torch_em.transform.get_raw_transform()
    channelwise_raw_transform = ChannelWiseRawTransform(base_transform)
    drop_channel = DropChannel(channel = 1)

    # get configurations
    has_sample_mask = sample_mask_paths is not None
    has_background_mask = background_mask_paths is not None

    # stack tomograms and masks and write to temp files to use as input to RawDataset()    
    stacked_paths = []

    # Case 1: both sample masks and background masks are provided, e.g., for the train data loader
    if has_sample_mask and has_background_mask: 
        assert len(data_paths) == len(sample_mask_paths) == len(background_mask_paths), \
        f"Expected equal number of paths, got {len(data_paths)} data paths, {len(sample_mask_paths)} sample mask paths \
            and {len(background_mask_paths)} background mask paths."
    
        for i, (data_path, sample_mask_path, background_mask_path) in enumerate(zip(data_paths, sample_mask_paths, background_mask_paths)):
            if Path(data_path).suffix == ".h5":
                with h5py.File(data_path, "r") as f:
                    raw = f[raw_key][:]
            else:
                raw = read_mrc(data_path)[0]
            sample_mask = read_mrc(sample_mask_path)[0]
            background_mask = read_mrc(background_mask_path)[0]

            stacked_path = get_stacked_path([raw, sample_mask, background_mask])
            stacked_paths.append(stacked_path)

        # update variables for RawDataset()
        data_paths = tuple(stacked_paths)    
        raw_transform = ComposedTransform(channelwise_raw_transform, drop_channel)
        augmentations = (ChannelWiseAugmentations(), ChannelWiseAugmentations())
        sampler = ChannelSplitterSampler(sampler)
        with_channels = True 

    # Case 2: only sample masks are provided, e.g., for the validation data loader
    elif has_sample_mask:
        assert len(data_paths) == len(sample_mask_paths), \
        f"Expected equal number of paths, got {len(data_paths)} data paths and {len(sample_mask_paths)} sample mask paths."

        for i, (data_path, sample_mask_path) in enumerate(zip(data_paths, sample_mask_paths)):
            if Path(data_path).suffix == ".h5":
                with h5py.File(data_path, "r") as f:
                    raw = f[raw_key][:]
            else:
                raw = read_mrc(data_path)[0]
            sample_mask = read_mrc(sample_mask_path)[0]

            stacked_path = get_stacked_path([raw, sample_mask])
            stacked_paths.append(stacked_path)

        # update variables for RawDataset()
        data_paths = tuple(stacked_paths)    
        raw_transform = ComposedTransform(base_transform, drop_channel)
        augmentations = (weak_augmentations(), weak_augmentations())
        sampler = ChannelSplitterSampler(sampler)
        with_channels = True 

    # Case 3: only background masks are provided 
    elif has_background_mask:
        # TODO test case 3
        assert len(data_paths) == len(background_mask_paths), \
        f"Expected equal number of paths, got {len(data_paths)} data paths and {len(background_mask_paths)} background mask paths."

        for i, (data_path, background_mask_path) in enumerate(zip(data_paths, background_mask_paths)):
            if Path(data_path).suffix == ".h5":
                with h5py.File(data_path, "r") as f:
                    raw = f[raw_key][:]
            else:
                raw = read_mrc(data_path)[0]
            background_mask = read_mrc(background_mask_path)[0]

            stacked_path = get_stacked_path([raw, background_mask])
            stacked_paths.append(stacked_path)

        # update variables for RawDataset()
        data_paths = tuple(stacked_paths)    
        raw_transform = base_transform
        augmentations = (ChannelWiseAugmentations(), ChannelWiseAugmentations())
        sampler = None
        with_channels = True 

    # Case 4: neither mask is present, use default behavior
    else:
        if Path(data_paths[0]).suffix != ".h5":
            raise ValueError("Input filepaths must point to hdf5 files if sample masks and/or background masks are not provided.")
        raw_transform = base_transform
        augmentations = (weak_augmentations(), weak_augmentations())
        sampler = None
        with_channels = False
    
    _, ndim = _determine_ndim(patch_shape)
    transform = torch_em.transform.get_augmentations(ndim=ndim)

    if n_samples is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples / len(data_paths))

    datasets = [
        torch_em.data.RawDataset(path, raw_key, patch_shape, raw_transform, transform, roi=roi,
        n_samples=n_samples_per_ds, sampler=sampler, ndim=ndim, with_channels=with_channels, augmentations=augmentations)
        for path in data_paths
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=True)
    return loader

# TODO: use different paths for supervised and unsupervised training
# (We are currently not using this functionality directly, so this is not a high priority)
def semisupervised_training(
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
        val_paths: Filepaths to the hdf5 files for the validation data.
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
