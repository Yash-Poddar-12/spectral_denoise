"""
ResNet Threshold Model for 1D Spectral Denoising
=================================================
A 1D Residual Shrinkage Network that combines residual connections with
learnable soft-thresholding activations. This approach is inspired by
deep residual shrinkage networks (DRSN) and is well-suited for spectral
denoising because soft-thresholding naturally suppresses small noise-like
coefficients while preserving significant spectral features.

Architecture:
    - Head conv: projects input to feature space
    - N residual shrinkage blocks (RSB): each block applies two conv layers
      with learnable per-channel soft-thresholding instead of ReLU
    - Tail conv: projects features back to signal space
    - Residual learning: final output = noisy_input - predicted_noise
      so the network only needs to learn the noise component

Reference concept: Zhao et al. "Deep Residual Shrinkage Networks for Fault
Diagnosis" (2019) — adapted here for 1D spectral denoising.
"""

import torch
import torch.nn as nn


class SoftThreshold(nn.Module):
    """Learnable per-channel soft-thresholding activation.

    For each channel c, the threshold tau_c is a trainable parameter
    initialised to a small positive value. The forward pass computes:

        y = sign(x) * max(|x| - |tau_c|, 0)

    Using |tau_c| ensures the threshold is always non-negative regardless
    of the sign of the raw parameter, which improves stability.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        # Shape (1, C, 1) broadcasts over batch and spatial dimensions
        self.threshold = nn.Parameter(torch.full((1, num_channels, 1), 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = torch.abs(self.threshold)
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


class ResidualShrinkageBlock(nn.Module):
    """Residual block that uses soft-thresholding instead of ReLU.

    Structure (for a block with ``channels`` feature maps):
        x --> Conv1d --> BN --> SoftThreshold
          --> Conv1d --> BN --> SoftThreshold
          --> (+) skip connection
          --> output

    The skip connection is a plain identity because the number of channels
    is kept constant throughout all residual blocks.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.thresh1 = SoftThreshold(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.thresh2 = SoftThreshold(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.thresh1(self.bn1(self.conv1(x)))
        out = self.thresh2(self.bn2(self.conv2(out)))
        return out + identity


class ResNetThreshold1D(nn.Module):
    """1D ResNet with learnable soft-thresholding for spectral denoising.

    The model uses *residual learning*: rather than predicting the clean
    spectrum directly it predicts the noise component and subtracts it
    from the noisy input:

        output = noisy_input - F(noisy_input)

    This makes training easier because the network only needs to capture
    the relatively small noise signal rather than the full spectral shape.

    Args:
        in_channels (int): Number of input channels (1 for a single spectrum).
        hidden_channels (int): Feature-map width used throughout the blocks.
        num_blocks (int): Number of residual shrinkage blocks.
        kernel_size (int): Convolution kernel size within each block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        # Project input to feature space
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_channels),
        )

        # Stack of residual shrinkage blocks
        self.blocks = nn.Sequential(
            *[ResidualShrinkageBlock(hidden_channels, kernel_size) for _ in range(num_blocks)]
        )

        # Project back to signal space — predicts the noise component
        self.tail = nn.Conv1d(hidden_channels, in_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy input tensor of shape (B, 1, L).

        Returns:
            Denoised output tensor of shape (B, 1, L).
        """
        features = self.head(x)
        noise_pred = self.tail(self.blocks(features))
        # Residual learning: subtract predicted noise from noisy input
        return x - noise_pred
