import torch
import torch.nn as nn


class MaskedDiceLossPerSample(nn.Module):
    """Dice loss with a per-voxel loss mask, which is averaged over the samples of the batch.

    The target holds the actual targets, followed by a mask (or weight) for each of them, as produced by
    `synapse_net.training.transform.AugmentedMitoStateMaskTransform`. The dice is computed per sample,
    so that every patch counts equally regardless of how much foreground it contains.

    Args:
        eps: The epsilon value used to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        # Enables torch-em to rebuild the loss when loading a checkpoint.
        self.init_kwargs = {"eps": eps}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the masked dice loss.

        Args:
            prediction: The predictions of the network.
            target: The targets, with the loss mask in the second half of the channels.

        Returns:
            The loss value.
        """
        n_channels = prediction.size(1)
        if target.size(1) != 2 * n_channels:
            raise ValueError(f"Expected a target with {2 * n_channels} channels, got {target.size(1)}.")
        # With the square root a weight enters the numerator and the denominator of the dice linearly.
        mask = torch.sqrt(target[:, n_channels:])
        pred, tgt = prediction * mask, target[:, :n_channels] * mask

        spatial_dims = tuple(range(2, pred.dim()))
        numerator = (pred * tgt).sum(spatial_dims)
        denominator = (pred * pred).sum(spatial_dims) + (tgt * tgt).sum(spatial_dims)
        dice = (2.0 * numerator / denominator.clamp(min=self.eps)).mean(dim=0)
        return (1.0 - dice).sum()
