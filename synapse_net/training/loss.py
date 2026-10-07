"""Loss functions that read a per-voxel loss mask from the training target.

These losses expect a target with twice the number of channels of the prediction: the first half
holds the actual targets, the second half holds a per-channel mask (or weight) that says how much
each voxel contributes to the loss. Targets of this shape are produced by
`synapse_net.training.transform.MitoStateMaskTransform` and are used for cristae training, where
mitochondria without cristae annotations have to be excluded from the loss.

Carrying the mask in the target keeps the loss a plain two-argument `loss(prediction, target)`,
so it needs no special support from the trainer.
"""

import torch
import torch.nn as nn


def _split_target(prediction: torch.Tensor, target: torch.Tensor, name: str):
    n_channels = prediction.size(1)
    if target.size(1) != 2 * n_channels:
        raise ValueError(f"{name} expects a target with {2 * n_channels} channels, got {target.size(1)}.")
    # The square root is applied so that a continuous weight w enters both the numerator and the
    # denominator of the dice linearly. For a binary mask this is the identity, so an unweighted
    # run is not affected by it.
    mask = torch.sqrt(target[:, n_channels:])
    return prediction * mask, target[:, :n_channels] * mask


class MaskedDiceLoss(nn.Module):
    """Dice loss that reads a per-channel loss mask from the second half of the target.

    The dice is computed per channel over the full batch (ratio of sums). See
    `MaskedDiceLossPerSample` for the variant that averages the dice over the batch instead.

    Args:
        eps: The epsilon value used to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        # Enables torch-em to serialize this loss into the checkpoint, so that the checkpoint can be
        # deserialized again without having to reconstruct the loss by hand.
        self.init_kwargs = {"eps": eps}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the masked dice loss.

        Args:
            prediction: The predictions of the network.
            target: The targets, with the loss mask in the second half of the channels.

        Returns:
            The loss value.
        """
        pred, tgt = _split_target(prediction, target, "MaskedDiceLoss")
        # Flatten to [C, B * spatial] so that the dice is computed per channel across the batch,
        # matching the reduction of torch_em's DiceLoss.
        n_channels = pred.size(1)
        pred = pred.permute(1, 0, *range(2, pred.dim())).reshape(n_channels, -1)
        tgt = tgt.permute(1, 0, *range(2, tgt.dim())).reshape(n_channels, -1)
        numerator = (pred * tgt).sum(-1)
        denominator = (pred * pred).sum(-1) + (tgt * tgt).sum(-1)
        return (1.0 - 2.0 * numerator / denominator.clamp(min=self.eps)).sum()


class MaskedDiceLossPerSample(nn.Module):
    """Masked dice loss that averages the dice over the batch instead of pooling it.

    The dice is computed for each element of the batch, reducing over the spatial dimensions only,
    and the results are then averaged (mean of ratios). Compared to `MaskedDiceLoss` (ratio of sums)
    this weights every patch equally, regardless of how many foreground voxels it contains, which
    counters the class imbalance of sparse structures such as cristae.

    Args:
        eps: The epsilon value used to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        # Enables torch-em to serialize this loss into the checkpoint, so that the checkpoint can be
        # deserialized again without having to reconstruct the loss by hand.
        self.init_kwargs = {"eps": eps}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the per-sample masked dice loss.

        Args:
            prediction: The predictions of the network.
            target: The targets, with the loss mask in the second half of the channels.

        Returns:
            The loss value.
        """
        pred, tgt = _split_target(prediction, target, "MaskedDiceLossPerSample")
        spatial_dims = tuple(range(2, pred.dim()))
        numerator = (pred * tgt).sum(spatial_dims)
        denominator = (pred * pred).sum(spatial_dims) + (tgt * tgt).sum(spatial_dims)
        dice = (2.0 * numerator / denominator.clamp(min=self.eps)).mean(dim=0)
        return (1.0 - dice).sum()
