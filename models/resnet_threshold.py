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


class DynamicSoftThreshold(nn.Module):
    """Dynamic per-sample, per-channel soft-thresholding activation for DRSN.

    Computes thresholds dynamically for each sample in the batch using a
    Squeeze-and-Excitation style sub-network. This allows the network to
    adaptively estimate the noise level per channel per signal!
    """

    def __init__(self, channels: int):
        super().__init__()
        # Squeeze-and-Excitation sub-network to generate scaling weights
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        # 1. Global Average Pooling -> (B, C)
        gap = abs_x.mean(dim=2)
        
        # 2. Generate scales -> (B, C)
        scales = self.fc(gap)
        
        # 3. Compute dynamic thresholds: tau = scales * gap -> (B, C, 1)
        tau = (scales * gap).unsqueeze(2)
        
        # 4. Soft thresholding: y = sign(x) * max(|x| - tau, 0)
        return torch.sign(x) * torch.clamp(abs_x - tau, min=0.0)


class ResidualShrinkageBlock(nn.Module):
    """Residual block with dynamic soft-thresholding."""

    def __init__(self, channels: int, kernel_size: int = 15):
        super().__init__()
        padding = kernel_size // 2
        
        # Block 1
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.thresh1 = DynamicSoftThreshold(channels)

        # Block 2
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.thresh2 = DynamicSoftThreshold(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.thresh1(self.bn1(self.conv1(x)))
        out = self.thresh2(self.bn2(self.conv2(out)))
        return out + x


class ResNetThreshold1D(nn.Module):
    """1D Deep Residual Shrinkage Network (DRSN) for spectral denoising.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Feature-map width used throughout the blocks.
        num_blocks (int): Number of residual shrinkage blocks.
        kernel_size (int): Convolution kernel size. Set to 7 for wider receptive fields.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        num_blocks: int = 12,
        kernel_size: int = 15,
    ):
        super().__init__()
        # Initial feature projection
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True) # Added initial activation to help map raw signal
        )

        # Stack of dynamic DRNS blocks
        self.blocks = nn.Sequential(
            *[ResidualShrinkageBlock(hidden_channels, kernel_size) for _ in range(num_blocks)]
        )

        # Reconstruct noise profile
        self.tail = nn.Conv1d(hidden_channels, in_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        noise_pred = self.tail(self.blocks(features))
        # Residual learning: predict noise and subtract
        return x - noise_pred
