from typing import Optional, Tuple, Union

import torch
import torch_em
from torch_em.data import RawDatasetWithMasks


def _adjust_patch_shape(data_shape, patch_shape):
    # If data is 2D and patch_shape is 3D, drop the extra dimension in patch_shape
    if data_shape == 2 and len(patch_shape) == 3:
        return patch_shape[1:]  # Remove the leading dimension in patch_shape
    return patch_shape  # Return the original patch_shape for 3D data


def _determine_ndim(patch_shape):
    # Check for 2D or 3D training
    try:
        z, y, x = patch_shape
    except ValueError:
        y, x = patch_shape
        z = 1
    is_2d = z == 1
    ndim = 2 if is_2d else 3
    return is_2d, ndim


def get_supervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    add_boundary_transform: bool = True,
    label_dtype=torch.float32,
    rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[Union[callable, bool]] = None,
    ignore_label: Optional[int] = None,
    label_transform: Optional[callable] = None,
    label_paths: Optional[Tuple[str]] = None,
    raw_transform: Optional[callable] = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for supervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        add_boundary_transform: Whether to add a boundary channel to the training data.
        label_dtype: The datatype of the labels returned by the dataloader.
        rois: Optional region of interests for training.
        sampler: Optional sampler to accept or reject patches for training. 
            By default a minimum instance sampler will be used, pass `False` to disable.
        ignore_label: Ignore label in the ground-truth. The areas marked by this label will be
            ignored in the loss computation. By default this option is not used.
        label_transform: Label transform that is applied to the segmentation to compute the targets.
            If no label transform is passed (the default) a boundary transform is used.
        label_paths: Optional paths containing the labels / annotations for training.
            If not given, the labels are expected to be contained in the `data_paths`.
        raw_transform: Normalization applied to the raw data. By default the torch_em standardization is used.
        loader_kwargs: Additional keyword arguments for the dataloader.

    Returns:
        The PyTorch dataloader.
    """
    _, ndim = _determine_ndim(patch_shape)
    if label_transform is not None:  # A specific label transform was passed, do nothing.
        pass
    elif add_boundary_transform:
        if ignore_label is None:
            label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
        else:
            label_transform = torch_em.transform.label.BoundaryTransformWithIgnoreLabel(
                add_binary_target=True, ignore_label=ignore_label
            )

    else:
        if ignore_label is not None:
            raise NotImplementedError
        label_transform = torch_em.transform.label.connected_components

    if ndim == 2:
        adjusted_patch_shape = _adjust_patch_shape(ndim, patch_shape)
        transform = torch_em.transform.Compose(
            torch_em.transform.PadIfNecessary(adjusted_patch_shape), torch_em.transform.get_augmentations(2)
        )
    else:
        transform = torch_em.transform.Compose(
            torch_em.transform.PadIfNecessary(patch_shape), torch_em.transform.get_augmentations(3)
        )

    num_workers = loader_kwargs.pop("num_workers", 4 * batch_size)
    shuffle = loader_kwargs.pop("shuffle", True)

    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)
    elif sampler is False:
        sampler = None

    if label_paths is None:
        label_paths = data_paths
    elif len(label_paths) != len(data_paths):
        raise ValueError(f"Data paths and label paths don't match: {len(data_paths)} != {len(label_paths)}")

    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        label_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape, ndim=ndim,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=shuffle, n_samples=n_samples,
        label_dtype=label_dtype, rois=rois, raw_transform=raw_transform, **loader_kwargs,
    )
    return loader


def get_unsupervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    sample_mask_paths: Optional[Tuple[str]] = None,
    sample_mask_key: Optional[str] = None,
    bg_mask_paths: Optional[Tuple[str]] = None,
    bg_mask_key: Optional[str] = None,
    sampler: Optional[callable] = None,
    exclude_top_and_bottom: bool = False,
    raw_transform: Optional[callable] = None,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        sample_mask_paths: The filepaths to the corresponding sample masks for each tomogram.
        sample_mask_key: The key to the sample mask dataset inside each file.
        bg_mask_paths: The filepaths to the background masks for each tomogram.
        bg_mask_key: The key to the background mask dataset inside each file.
        sampler: Optional sampler to accept or reject patches for training. 
        exclude_top_and_bottom: Whether to exclude the five top and bottom slices to
            avoid artifacts at the border of tomograms.
        raw_transform: Normalization applied to the raw data. By default the torch_em standardization is used.

    Returns:
        The PyTorch dataloader.
    """
    if exclude_top_and_bottom:
        roi = (slice(5, -5), slice(None), slice(None))
    else:
        roi = None

    if sample_mask_paths is not None:
        assert len(data_paths) == len(sample_mask_paths), \
            f"Expected equal number of data_paths and sample_mask_paths, got {len(data_paths)} and {len(sample_mask_paths)}."
    if bg_mask_paths is not None:
        assert len(data_paths) == len(bg_mask_paths), \
            f"Expected equal number of data_paths and bg_mask_paths, got {len(data_paths)} and {len(bg_mask_paths)}."

    _, ndim = _determine_ndim(patch_shape)
    if raw_transform is None:
        raw_transform = torch_em.transform.get_raw_transform()
    transform = torch_em.transform.get_augmentations(ndim=ndim)

    if n_samples is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples / len(data_paths))

    datasets = [
        RawDatasetWithMasks(
            raw_path=data_path,
            raw_key=raw_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            transform=transform,
            roi=roi,
            n_samples=n_samples_per_ds,
            sampler=sampler,
            ndim=ndim,
            augmentations=None,
            sample_mask_path=sample_mask_paths[i] if sample_mask_paths is not None else None,
            sample_mask_key=sample_mask_key,
            bg_mask_path=bg_mask_paths[i] if bg_mask_paths is not None else None,
            bg_mask_key=bg_mask_key,
        )
        for i, data_path in enumerate(data_paths)
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=True)
    return loader


