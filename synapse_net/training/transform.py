import numpy as np
import torch
import torch_em
from bioimage_cpp.distance import distance_transform
from torch_em.transform.label import labels_to_binary
from scipy.ndimage import distance_transform_edt

# The voxel size of the cristae training data, in nanometer.
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
    """Standardize one channel of a multi-channel input and leave the other channels unchanged.

    Args:
        raw: The input data, with the channels in the first axis.
        channel: The channel to standardize.

    Returns:
        The input data with the given channel standardized.
    """
    if raw.ndim != 4:
        raise ValueError(f"Expect input data with 4 dimensions (channels, z, y, x), got {raw.ndim}.")
    raw = np.float32(raw)
    raw[channel] = torch_em.transform.raw.standardize(raw[channel])
    return raw


def _membrane_weight(state, foreground, voxel_size, band_nm, offset_nm, w_pos, w_neg, exclude_state_value):
    # Zero in the mitochondria without cristae annotations, one everywhere else (including the background).
    weight = (np.abs(state - exclude_state_value) >= 0.5).astype(np.float32)
    mito = state == 1
    if (w_pos == 1.0 and w_neg == 1.0) or not mito.any():
        return weight

    # The membrane is approximated by the surface of the annotated mitochondria. Padding in 'edge' mode lets a
    # mitochondrion that is cut by the patch border continue, instead of creating a band through its lumen.
    upper = offset_nm + band_nm
    pad = int(np.ceil(upper / min(voxel_size))) + 2
    distances = distance_transform(np.pad(mito, pad, mode="edge"), sampling=list(voxel_size), number_of_threads=1)
    distances = distances[tuple(slice(pad, -pad) for _ in range(mito.ndim))]
    band = mito & (distances > offset_nm) & (distances <= upper)

    foreground = foreground > 0
    weight[band & foreground] *= np.float32(w_pos)
    weight[band & ~foreground] *= np.float32(w_neg)
    return weight


class AugmentedMitoStateMaskTransform:
    """Joint transform that augments the data and appends the loss mask for cristae training to the labels.

    The loss mask is computed from the mitochondria state channel of the augmented raw data. It is zero in the
    mitochondria without cristae annotations (the state `exclude_state_value`) and one elsewhere. Inside of a band
    at the membrane of the annotated mitochondria (state 1) it is `w_pos` for cristae and `w_neg` for the rest.
    The labels are returned with twice their channels, as expected by
    `synapse_net.training.loss.MaskedDiceLossPerSample`.

    Args:
        augmentation: The joint augmentation, e.g. `torch_em.transform.get_augmentations(3)`.
        mito_channel: The channel of the raw data that holds the mitochondria state.
        exclude_state_value: The state of the mitochondria without cristae annotations.
        band_nm: The thickness of the membrane band in nanometer.
        offset_nm: Start the band this far inside the membrane.
        w_pos: The weight for cristae voxels inside the band.
        w_neg: The weight for the other voxels inside the band.
        voxel_size: The voxel size in nanometer, in the order (z, y, x).
    """

    def __init__(
        self,
        augmentation,
        mito_channel: int = 1,
        exclude_state_value: float = 2.0,
        band_nm: float = 12.0,
        offset_nm: float = 0.0,
        w_pos: float = 1.0,
        w_neg: float = 1.0,
        voxel_size=CRISTAE_VOXEL_SIZE,
    ):
        self.augmentation = augmentation
        self.mito_channel = mito_channel
        self.exclude_state_value = float(exclude_state_value)
        self.band_nm = float(band_nm)
        self.offset_nm = float(offset_nm)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)
        self.voxel_size = tuple(float(vs) for vs in voxel_size)

    def __call__(self, raw, labels):
        """Augment the raw data and labels, then append the loss mask to the labels.

        Args:
            raw: The raw data, with the mitochondria state in `mito_channel`.
            labels: The labels produced by the label transform, with the binary cristae in the first channel.

        Returns:
            The augmented raw data.
            The augmented labels with the loss mask appended to the channel axis.
        """
        # The augmentations return the data with a leading batch axis.
        raw, labels = self.augmentation(raw, labels)
        raw, labels = torch.as_tensor(raw), torch.as_tensor(labels)
        state = raw[:, self.mito_channel].float().numpy()
        foreground = labels[:, 0].numpy()
        mask = np.stack([
            _membrane_weight(s, f, self.voxel_size, self.band_nm, self.offset_nm, self.w_pos, self.w_neg,
                             self.exclude_state_value)
            for s, f in zip(state, foreground)
        ])
        mask = torch.as_tensor(mask).to(labels.dtype)
        masks = mask.unsqueeze(1).repeat(1, labels.shape[1], *([1] * (labels.dim() - 2)))
        return raw, torch.cat([labels, masks], dim=1)
