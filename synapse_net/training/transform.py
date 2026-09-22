"""Transformations applied to the data and labels during training.
"""

import numpy as np
import torch
import torch_em
from torch_em.transform.label import labels_to_binary
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as connected_components

# The voxel size of the cristae training data, in nanometer. Used when no voxel size is given.
CRISTAE_VOXEL_SIZE = (1.74, 1.74, 1.74)


class AZDistanceLabelTransform:
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance

    def __call__(self, input_):
        binary_target = labels_to_binary(input_).astype("float32")
        if binary_target.sum() == 0:
            distances = np.ones_like(binary_target, dtype="float32")
        else:
            distances = distance_transform_edt(binary_target == 0)
            distances = np.clip(distances, 0.0, self.max_distance)
            distances /= self.max_distance
        return np.stack([binary_target, distances])


def standardize_channel(raw: np.ndarray, channel: int = 0) -> np.ndarray:
    """Standardize a single channel of a multi-channel input, leaving the other channels unchanged.

    This is used for training on data where one channel holds the image data and another one holds a
    semantic mask, which has to keep its integer values so that it can be compared against them.

    Args:
        raw: The multi-channel input data, with the channels in the first axis.
        channel: The channel to standardize.

    Returns:
        The input data with the given channel standardized.

    Raises:
        ValueError: If the input is not 4D or if the channel does not exist.
    """
    if raw.ndim != 4:
        raise ValueError(f"Expect input data with 4 dimensions (channels, z, y, x), got {raw.ndim}.")
    if channel >= raw.shape[0]:
        raise ValueError(f"Invalid channel {channel} for input data with {raw.shape[0]} channels.")
    raw = np.float32(raw)
    raw[channel] = torch_em.transform.raw.standardize(raw[channel])
    return raw


def normalize_channel(raw: np.ndarray, channel: int = 0, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Percentile-normalize a single channel of a multi-channel input, leaving the others unchanged.

    See `standardize_channel` for why only one channel is normalized.

    Args:
        raw: The multi-channel input data, with the channels in the first axis.
        channel: The channel to normalize.
        lower: The lower percentile.
        upper: The upper percentile.

    Returns:
        The input data with the given channel normalized to the range [0, 1].

    Raises:
        ValueError: If the input is not 4D or if the channel does not exist.
    """
    if raw.ndim != 4:
        raise ValueError(f"Expect input data with 4 dimensions (channels, z, y, x), got {raw.ndim}.")
    if channel >= raw.shape[0]:
        raise ValueError(f"Invalid channel {channel} for input data with {raw.shape[0]} channels.")
    raw = np.float32(raw)
    normalized = torch_em.transform.raw.normalize_percentile(raw[channel], lower=lower, upper=upper)
    raw[channel] = np.clip(normalized, 0, 1)
    return raw


def remove_white_patches(raw: np.ndarray, min_size: int = 20, value: int = 255) -> np.ndarray:
    """Set the connected components of saturated voxels in a volume EM block to zero.

    Volume EM blocks that were cut out of a larger FIB-SEM volume carry white filler borders where the
    cutout extends past the imaged region. They take up a large part of some blocks, so they skew the
    percentile normalization of the data around them. Electron tomograms do not have these borders,
    this is only meant for volume EM data.

    Small saturated components are kept, because they are saturated sample structure rather than
    filler. The components are determined with full connectivity, so that filler that is only
    connected diagonally still counts as one component.

    Args:
        raw: The volume, with the values of the filler given by `value`.
        min_size: The minimal size (in voxels) of a component that is removed.
        value: The value that marks the filler.

    Returns:
        The volume with the filler set to zero. The input is returned unchanged if it holds no filler.

    Raises:
        ValueError: If the volume is not 2d or 3d, or if it does not have an integer dtype.
    """
    if raw.ndim not in (2, 3):
        raise ValueError(f"Expect 2d or 3d input data, got {raw.ndim} dimensions.")
    # The filler is identified by an exact value, so this has to run on the unnormalized data.
    # Normalized or standardized data would silently contain no filler at all.
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(
            f"Expect input data with an integer dtype, got {raw.dtype}. This transform has to be applied "
            "to the raw data, before it is normalized."
        )

    # This runs once per patch in every dataloader worker, and most patches do not touch the border.
    mask = raw == value
    if not mask.any():
        return raw

    components = connected_components(mask)
    sizes = np.bincount(components.ravel())
    filler_ids = np.where(sizes >= min_size)[0]
    filler_ids = filler_ids[filler_ids != 0]
    if filler_ids.size == 0:
        return raw

    raw = raw.copy()
    raw[np.isin(components, filler_ids)] = 0
    return raw


class RemoveWhitePatchesAndNormalize:
    """Remove the white filler borders of a volume EM block, then percentile-normalize it.

    This is the **training** raw transform of SynapseNet's volume EM mitochondria model, see
    `synapse_net.training.mitochondria_vol_em`. It is applied per patch, to the unnormalized data that
    the dataset reads. It is a class rather than a function so that it can be pickled by reference:
    the dataloader workers and the training checkpoint both require that.

    Do not pass it as the `preprocess` of an inference function. `synapse_net.inference.util.get_prediction`
    standardizes a numpy input volume before `preprocess` runs, so the filler would no longer have the
    value that identifies it. At inference, call `remove_white_patches` on the whole volume first and
    then normalize with `torch_em.transform.raw.normalize_percentile`.

    Args:
        min_size: The minimal size (in voxels) of a filler component that is removed.
        lower: The lower percentile for the normalization.
        upper: The upper percentile for the normalization.
    """

    def __init__(self, min_size: int = 20, lower: float = 1.0, upper: float = 99.0):
        self.min_size = min_size
        self.lower = lower
        self.upper = upper

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        raw = remove_white_patches(raw, min_size=self.min_size)
        return torch_em.transform.raw.normalize_percentile(raw, lower=self.lower, upper=self.upper)


def _fast_distance_transform(mask, sampling):
    # This runs once per patch in every dataloader worker, so it is worth using the faster
    # implementation from bioimage-cpp when it is available (measured ~5x faster than scipy on a
    # padded 32 x 256 x 256 patch, agreeing to within float32 rounding).
    # It is called single-threaded on purpose: the dataloader workers are already parallel.
    try:
        from bioimage_cpp.distance import distance_transform
    except ImportError:
        return distance_transform_edt(mask, sampling=sampling)
    return distance_transform(mask, sampling=list(sampling), number_of_threads=1)


def proximity_band(
    mito_mask: np.ndarray,
    voxel_size=CRISTAE_VOXEL_SIZE,
    band_nm: float = 12.0,
    offset_nm: float = 0.0,
) -> np.ndarray:
    """Compute the shell inside `mito_mask` with a distance to its surface in the given band.

    The shell contains the voxels with `offset_nm < distance <= offset_nm + band_nm`. Since the
    mitochondrial membrane is not annotated in the training data, the surface of the mitochondria
    mask is used to approximate it.

    This is meant to run on a training patch, so it corrects for the patch borders: a mitochondrion
    that is cut by a patch face would otherwise be seen as ending there, and the distance transform
    would introduce a membrane band through the lumen of the mitochondrion. Padding the mask in
    'edge' mode before the transform lets such a mitochondrion continue instead. Where the face is
    background, the padding does not change anything.

    Args:
        mito_mask: The binary mitochondria mask.
        voxel_size: The voxel size in nanometer, in the order (z, y, x).
        band_nm: The thickness of the shell in nanometer.
        offset_nm: Start the shell this far inside the membrane. Zero starts it at the mask surface.

    Returns:
        The binary shell, with the same shape as `mito_mask`.
    """
    mito_mask = np.asarray(mito_mask, dtype=bool)
    if not mito_mask.any():
        return np.zeros(mito_mask.shape, dtype=bool)

    lower = float(offset_nm)
    upper = lower + float(band_nm)
    pad = int(np.ceil(upper / float(min(voxel_size)))) + 2
    distances = _fast_distance_transform(np.pad(mito_mask, pad, mode="edge"), voxel_size)
    distances = distances[tuple(slice(pad, -pad) for _ in range(mito_mask.ndim))]
    return mito_mask & (distances > lower) & (distances <= upper)


def membrane_proximity_weight(
    mito_state: np.ndarray,
    gt_foreground: np.ndarray,
    voxel_size=CRISTAE_VOXEL_SIZE,
    band_nm: float = 12.0,
    offset_nm: float = 0.0,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    annotated_state: int = 1,
    exclude_state_value: float = 2.0,
) -> np.ndarray:
    """Compute a per-voxel loss weight that emphasizes the cristae close to the mitochondria membrane.

    This generalizes the binary loss mask used for cristae training::

        w = 0.0    where the state is `exclude_state_value`, i.e. an unannotated mitochondrion
        w = 1.0    everywhere else
        w = w_pos  inside the membrane band and in the cristae ground-truth
        w = w_neg  inside the membrane band and outside of the cristae ground-truth

    Splitting the weight into `w_pos` and `w_neg` is the point of this function: a single band weight
    would amplify the cristae and the membrane equally. `w_pos` pushes the model towards predicting
    cristae at the junction with the membrane, `w_neg` separately controls how strongly the membrane
    itself is pushed down.

    With `w_pos == w_neg == 1.0` the result is exactly the binary loss mask, so the unweighted recipe
    is not affected by this function and no distance transform is computed.

    Args:
        mito_state: The semantic mitochondria state, where 0 is background, 1 is a mitochondrion with
            cristae annotations, and 2 is a mitochondrion without cristae annotations.
        gt_foreground: The binary cristae ground-truth, with the same shape as `mito_state`.
        voxel_size: The voxel size in nanometer, in the order (z, y, x).
        band_nm: The thickness of the membrane band in nanometer.
        offset_nm: Start the band this far inside the membrane.
        w_pos: The weight for cristae voxels inside the band.
        w_neg: The weight for non-cristae voxels inside the band.
        annotated_state: The state value of mitochondria that carry cristae annotations.
        exclude_state_value: The state value that is excluded from the loss.

    Returns:
        The per-voxel weight, with the same shape as `mito_state`.
    """
    mito_state = np.asarray(mito_state)
    # One where the state is NOT excluded from the loss. Note that this keeps the background, so that
    # the model still learns that there are no cristae outside of mitochondria.
    weight = (np.abs(mito_state - float(exclude_state_value)) >= 0.5).astype(np.float32)

    w_pos, w_neg = float(w_pos), float(w_neg)
    if w_pos == 1.0 and w_neg == 1.0:
        return weight

    band = proximity_band(mito_state == annotated_state, voxel_size, band_nm, offset_nm)
    if not band.any():
        return weight

    foreground = np.asarray(gt_foreground) > 0
    # Multiply rather than assign, so that excluded voxels keep a weight of zero.
    if w_pos != 1.0:
        weight[band & foreground] *= np.float32(w_pos)
    if w_neg != 1.0:
        weight[band & ~foreground] *= np.float32(w_neg)
    return weight


class MitoStateMaskTransform:
    """Joint transform that appends the cristae loss mask to the labels.

    After the label transform has produced labels of shape (n_channels, z, y, x), this appends
    n_channels mask channels that encode where the loss should be computed, resulting in labels of
    shape (2 * n_channels, z, y, x) as expected by
    `synapse_net.training.loss.MaskedDiceLossPerSample`.

    Voxels where the mitochondria state channel of the raw data equals `exclude_state_value` are
    masked out, so that the network is not penalized for predicting cristae inside mitochondria that
    carry no cristae annotations. All other voxels, including the background, stay active.

    The mask can also become a weight map that emphasizes the membrane proximity band, see
    `membrane_proximity_weight`. With the default `w_pos == w_neg == 1.0` it is exactly the binary
    mask. Note that this transform replaces the augmentations in the dataloader; use
    `AugmentedMitoStateMaskTransform` to keep them.

    Args:
        mito_channel: The channel of the raw data that holds the mitochondria state.
        exclude_state_value: The state value that is excluded from the loss.
        band_nm: The thickness of the membrane band in nanometer.
        offset_nm: Start the band this far inside the membrane.
        w_pos: The weight for cristae voxels inside the band.
        w_neg: The weight for non-cristae voxels inside the band.
        voxel_size: The voxel size in nanometer, in the order (z, y, x).
    """

    def __init__(
        self,
        mito_channel: int = 1,
        exclude_state_value: float = 2.0,
        band_nm: float = 12.0,
        offset_nm: float = 0.0,
        w_pos: float = 1.0,
        w_neg: float = 1.0,
        voxel_size=CRISTAE_VOXEL_SIZE,
    ):
        self.mito_channel = mito_channel
        self.exclude_state_value = exclude_state_value
        self.band_nm = float(band_nm)
        self.offset_nm = float(offset_nm)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)
        self.voxel_size = tuple(float(vs) for vs in voxel_size)

    @property
    def weighted(self) -> bool:
        """Whether a membrane proximity weight is applied instead of the plain binary mask."""
        return self.w_pos != 1.0 or self.w_neg != 1.0

    def _weight(self, mito_state, gt_foreground):
        return membrane_proximity_weight(
            mito_state, gt_foreground, voxel_size=self.voxel_size, band_nm=self.band_nm,
            offset_nm=self.offset_nm, w_pos=self.w_pos, w_neg=self.w_neg,
            exclude_state_value=self.exclude_state_value,
        )

    def __call__(self, raw, labels):
        """Append the loss mask to the labels.

        Args:
            raw: The raw data, with the mitochondria state in `mito_channel`.
            labels: The labels produced by the label transform.

        Returns:
            The unchanged raw data.
            The labels with the loss mask appended to the channel axis.
        """
        mito_state = raw[self.mito_channel]
        if self.weighted:
            # The first label channel is the binary cristae channel from the boundary transform.
            mask = self._weight(mito_state, labels[0])
        else:
            mask = (np.abs(mito_state - self.exclude_state_value) >= 0.5).astype(np.float32)
        masks = np.stack([mask] * labels.shape[0], axis=0)
        return raw, np.concatenate([labels, masks], axis=0)


class AugmentedMitoStateMaskTransform(MitoStateMaskTransform):
    """`MitoStateMaskTransform` that applies an augmentation before computing the loss mask.

    A joint transform occupies the slot of the dataloader that otherwise holds the augmentations, so
    `MitoStateMaskTransform` on its own trains without any augmentation. This class restores them: it
    augments the raw data and labels first and then computes the loss mask from the augmented state
    channel, so that the mask matches the data the network sees.

    Args:
        augmentation: The joint augmentation for the raw data and labels. Pass
            `torch_em.transform.get_augmentations(ndim)`, which is what the dataloader uses by default.
        kwargs: Additional arguments for `MitoStateMaskTransform`.
    """

    def __init__(self, augmentation, **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation

    def _weight_torch(self, mito_state, gt_foreground, channel_axis):
        # The default augmentations are flips only, so the state channel stays integral and the
        # distance transform runs on the same geometry the network sees.
        state = mito_state.detach().cpu().numpy()
        foreground = gt_foreground.detach().cpu().numpy()
        if channel_axis == 1:  # There is a leading batch axis, so the transform has to run per sample.
            return np.stack([self._weight(state[i], foreground[i]) for i in range(state.shape[0])], axis=0)
        return self._weight(state, foreground)

    def __call__(self, raw, labels):
        """Augment the raw data and labels, then append the loss mask to the labels.

        Args:
            raw: The raw data, with the mitochondria state in `mito_channel`.
            labels: The labels produced by the label transform.

        Returns:
            The augmented raw data.
            The augmented labels with the loss mask appended to the channel axis.
        """
        raw, labels = self.augmentation(raw, labels)
        raw, labels = torch.as_tensor(raw), torch.as_tensor(labels)
        # The augmentations return the data with a leading batch axis, so the channel axis is one.
        # Support data without it as well, where the channel axis is zero.
        channel_axis = 1 if raw.dim() >= 5 else 0
        mito_state = raw.narrow(channel_axis, self.mito_channel, 1).squeeze(channel_axis).float()
        if self.weighted:
            # The first label channel is the binary cristae channel from the boundary transform.
            foreground = labels.narrow(channel_axis, 0, 1).squeeze(channel_axis)
            mask = torch.as_tensor(self._weight_torch(mito_state, foreground, channel_axis)).to(labels.dtype)
        else:
            mask = (torch.abs(mito_state - self.exclude_state_value) >= 0.5).to(labels.dtype)
        repeats = [1] * labels.dim()
        repeats[channel_axis] = labels.shape[channel_axis]
        masks = mask.unsqueeze(channel_axis).repeat(*repeats)
        return raw, torch.cat([labels, masks], dim=channel_axis)
