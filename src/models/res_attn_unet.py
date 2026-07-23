"""3D Residual Attention U-Net for MRI→PET synthesis.

Architecture:
- Encoder: 4 levels of residual blocks with max-pooling
- Bottleneck: residual block
- Decoder: 4 levels of transposed conv + skip connections with attention gates
- Output: 1×1×1 conv → synthesized PET

Reference:
  Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas." MIDL 2018.
  He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.base import BaseModel


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ResidualBlock3D(nn.Module):
    """3D Residual block: Conv-BN-ReLU-Conv-BN + skip connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout probability (applied after second conv).
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 1×1 projection if channel dims differ
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection.
        """
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class AttentionGate3D(nn.Module):
    """3D Additive Attention Gate.

    Computes soft attention coefficients from skip connection (g)
    and decoder feature map (x) to suppress irrelevant activations.

    Args:
        f_g: Number of channels in the gating signal.
        f_l: Number of channels in the skip connection.
        f_int: Number of intermediate channels.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, 1, bias=True),
            nn.BatchNorm3d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, 1, stride=1, bias=True),
            nn.BatchNorm3d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """Compute gated attention.

        Args:
            g: Gating signal from decoder (coarser scale).
            x: Skip connection features from encoder.

        Returns:
            Attention-weighted skip features.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block: ResidualBlock + MaxPool.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.res = ResidualBlock3D(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (pooled_output, skip_features).
        """
        skip = self.res(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Decoder block: TransposeConv + AttentionGate + ResidualBlock.

    Args:
        in_ch: Input channels (from previous decoder level).
        skip_ch: Skip connection channels.
        out_ch: Output channels.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate3D(f_g=in_ch // 2, f_l=skip_ch, f_int=out_ch // 2)
        self.res = ResidualBlock3D(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input from previous decoder level.
            skip: Skip connection from encoder.

        Returns:
            Decoded feature tensor.
        """
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ─── Full Model ───────────────────────────────────────────────────────────────

class ResAttnUNet3D(BaseModel):
    """3D Residual Attention U-Net for MRI→PET synthesis.

    Args:
        in_channels: Number of input channels (default 1 for MRI).
        out_channels: Number of output channels (default 1 for PET).
        features: Channel sizes for each encoder level.
        dropout: Dropout probability in residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._features = features
        self._dropout = dropout

        # Encoder
        self.enc1 = EncoderBlock(in_channels, features[0], dropout)
        self.enc2 = EncoderBlock(features[0], features[1], dropout)
        self.enc3 = EncoderBlock(features[1], features[2], dropout)
        self.enc4 = EncoderBlock(features[2], features[3], dropout)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(features[3], features[4], dropout)

        # Decoder
        self.dec4 = DecoderBlock(features[4], features[3], features[3], dropout)
        self.dec3 = DecoderBlock(features[3], features[2], features[2], dropout)
        self.dec2 = DecoderBlock(features[2], features[1], features[1], dropout)
        self.dec1 = DecoderBlock(features[1], features[0], features[0], dropout)

        # Output
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, mri: Tensor) -> Tensor:
        """Synthesize PET from MRI.

        Args:
            mri: Input MRI tensor of shape (B, 1, D, H, W).

        Returns:
            Synthesized PET tensor of shape (B, 1, D, H, W).
        """
        # Encode
        x1, s1 = self.enc1(mri)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck
        x = self.bottleneck(x4)

        # Decode
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.output_conv(x)

    def get_config(self) -> dict:
        """Return serializable model configuration.

        Returns:
            Dict describing model architecture.
        """
        return {
            "name": "res_attn_unet",
            "in_channels": self._in_channels,
            "out_channels": self._out_channels,
            "features": self._features,
            "dropout": self._dropout,
            "num_parameters": self.get_num_parameters(),
        }
