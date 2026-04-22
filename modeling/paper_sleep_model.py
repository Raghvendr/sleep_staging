from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class PaperSleepStageModelConfig:
    num_classes: int = 4
    patch_samples: int = 256
    embedding_dim: int = 128
    local_blocks: int = 3
    local_kernel_size: int = 3
    temporal_blocks: int = 2
    temporal_kernel_size: int = 7
    dilations: tuple[int, ...] = (2, 4, 8, 16, 32)
    dropout: float = 0.2
    negative_slope: float = 0.01
    use_batch_norm: bool = False
    local_channels: tuple[int, ...] = field(default_factory=lambda: (16, 32, 64))


def _conv1d_same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


class LocalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        negative_slope: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        padding = _conv1d_same_padding(kernel_size=kernel_size, dilation=1)

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.block = nn.Sequential(*layers)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.residual_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        residual = self.residual_pool(residual)

        x = self.block(x)
        x = self.pool(x)
        return x + residual


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        dropout: float,
        negative_slope: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dilation in dilations:
            padding = _conv1d_same_padding(kernel_size=kernel_size, dilation=dilation)
            layers.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)
        x = self.dropout(x)
        return x + residual


class PaperSleepStageModel(nn.Module):
    """
    Paper-inspired model from:
    "Deep learning for automated sleep staging using instantaneous heart rate"

    Expected input shape:
    - batch x epochs x patch_samples
    Output shape:
    - batch x epochs x num_classes
    """

    def __init__(self, config: PaperSleepStageModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PaperSleepStageModelConfig()

        channels = self.config.local_channels
        if len(channels) != self.config.local_blocks:
            raise ValueError("local_channels length must equal local_blocks")

        local_blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            local_blocks.append(
                LocalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config.local_kernel_size,
                    negative_slope=self.config.negative_slope,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )
            in_channels = out_channels
        self.local_feature_extractor = nn.Sequential(*local_blocks)

        downsample_factor = 2 ** self.config.local_blocks
        reduced_patch_samples = self.config.patch_samples // downsample_factor
        if reduced_patch_samples <= 0:
            raise ValueError("patch_samples must remain positive after local downsampling")

        self.local_projection = nn.Linear(
            in_features=channels[-1] * reduced_patch_samples,
            out_features=self.config.embedding_dim,
        )

        temporal_blocks: list[nn.Module] = []
        for _ in range(self.config.temporal_blocks):
            temporal_blocks.append(
                DilatedResidualBlock(
                    channels=self.config.embedding_dim,
                    kernel_size=self.config.temporal_kernel_size,
                    dilations=self.config.dilations,
                    dropout=self.config.dropout,
                    negative_slope=self.config.negative_slope,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )
        self.temporal_feature_extractor = nn.Sequential(*temporal_blocks)
        self.classifier = nn.Conv1d(self.config.embedding_dim, self.config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_epochs, patch_samples = x.shape
        if patch_samples != self.config.patch_samples:
            raise ValueError(
                f"Expected patch_samples={self.config.patch_samples}, got {patch_samples}"
            )

        x = x.reshape(batch_size * num_epochs, 1, patch_samples)
        x = self.local_feature_extractor(x)
        x = x.reshape(batch_size * num_epochs, -1)
        x = self.local_projection(x)

        x = x.reshape(batch_size, num_epochs, self.config.embedding_dim)
        x = x.transpose(1, 2)
        x = self.temporal_feature_extractor(x)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        return x
