"""Combined loss function for MRI→PET synthesis.

Combines: L1 + 3D SSIM (+ optional perceptual, adversarial, gradient).
All weights configurable via LossConfig.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import LossConfig

logger = logging.getLogger(__name__)


# ─── L1 Loss (masked) ─────────────────────────────────────────────────────────

class MaskedL1Loss(nn.Module):
    """L1 loss computed only inside a brain mask (if provided).

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute masked L1 loss.

        Args:
            pred: Predicted PET tensor (B, 1, D, H, W).
            target: Ground truth PET tensor (B, 1, D, H, W).
            mask: Optional binary brain mask (B, 1, D, H, W).

        Returns:
            Scalar loss tensor.
        """
        diff = torch.abs(pred - target)
        if mask is not None:
            diff = diff * mask
            n = mask.sum().clamp(min=1)
            return diff.sum() / n
        return diff.mean() if self.reduction == "mean" else diff.sum()


# ─── 3D SSIM Loss ─────────────────────────────────────────────────────────────

def _gaussian_kernel_3d(window_size: int, sigma: float, channels: int) -> Tensor:
    """Create a 3D Gaussian kernel for SSIM computation.

    Args:
        window_size: Kernel size (odd number recommended).
        sigma: Gaussian sigma.
        channels: Number of channels.

    Returns:
        Normalized Gaussian kernel of shape (channels, 1, ws, ws, ws).
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(0)
    kernel_3d = kernel_1d.unsqueeze(-1).unsqueeze(-1) * kernel_1d.unsqueeze(-1) * kernel_1d
    kernel_3d = kernel_3d / kernel_3d.sum()
    return kernel_3d.expand(channels, 1, window_size, window_size, window_size).contiguous()


class SSIMLoss3D(nn.Module):
    """3D Structural Similarity Index (SSIM) loss.

    Returns 1 - SSIM so that minimizing the loss maximizes SSIM.

    Args:
        window_size: Gaussian kernel size.
        sigma: Gaussian sigma.
        data_range: Expected data range (e.g., 1.0 for normalized images).
        size_average: If True, average over the batch.
    """

    def __init__(
        self,
        window_size: int = 7,
        sigma: float = 1.5,
        data_range: float = 1.0,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        self._window: Optional[Tensor] = None

    def _get_window(self, channels: int, device: torch.device) -> Tensor:
        """Get or build the Gaussian kernel.

        Args:
            channels: Number of image channels.
            device: Target device.

        Returns:
            Gaussian kernel tensor.
        """
        if self._window is None or self._window.device != device:
            self._window = _gaussian_kernel_3d(self.window_size, self.sigma, channels).to(device)
        return self._window

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute 1 - SSIM loss.

        Args:
            pred: Predicted PET (B, 1, D, H, W).
            target: Ground truth PET (B, 1, D, H, W).
            mask: Optional brain mask (unused in windowed SSIM, kept for API consistency).

        Returns:
            Scalar loss (1 - mean SSIM).
        """
        channels = pred.shape[1]
        window = self._get_window(channels, pred.device)
        pad = self.window_size // 2

        mu1 = F.conv3d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv3d(target, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv3d(target * target, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv3d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        if self.size_average:
            return 1.0 - ssim_map.mean()
        return 1.0 - ssim_map.mean(dim=[1, 2, 3, 4])


# ─── Gradient Difference Loss ─────────────────────────────────────────────────

class GradientDifferenceLoss3D(nn.Module):
    """Gradient Difference Loss in 3D.

    Encourages sharpness by penalizing gradient magnitude differences.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute gradient difference loss.

        Args:
            pred: Predicted PET (B, 1, D, H, W).
            target: Ground truth PET (B, 1, D, H, W).

        Returns:
            Scalar loss.
        """
        def grad(x: Tensor) -> Tensor:
            gd = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
            gh = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
            gw = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
            return gd + gh + gw

        return torch.abs(grad(pred) - grad(target))


# ─── Combined Loss ────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Weighted combination of L1 + SSIM + gradient + optional adversarial.

    Args:
        cfg: Loss configuration with per-component weights.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.l1 = MaskedL1Loss() if cfg.l1_weight > 0 else None
        self.ssim = SSIMLoss3D() if cfg.ssim_weight > 0 else None
        self.grad = GradientDifferenceLoss3D() if cfg.gradient_weight > 0 else None

        active_components = [
            f"L1×{cfg.l1_weight}" if cfg.l1_weight > 0 else None,
            f"SSIM×{cfg.ssim_weight}" if cfg.ssim_weight > 0 else None,
            f"Gradient×{cfg.gradient_weight}" if cfg.gradient_weight > 0 else None,
            f"Adversarial×{cfg.adversarial_weight}" if cfg.adversarial_weight > 0 else None,
        ]
        logger.info("CombinedLoss: %s", " + ".join(c for c in active_components if c))

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        disc_loss: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: Synthesized PET (B, 1, D, H, W).
            target: Ground truth PET (B, 1, D, H, W).
            mask: Optional brain mask.
            disc_loss: Adversarial loss from discriminator (if GAN mode).

        Returns:
            Tuple of (total_loss_tensor, component_dict for logging).
        """
        total = torch.zeros(1, device=pred.device)
        components: dict[str, float] = {}

        if self.l1 is not None:
            loss_l1 = self.l1(pred, target, mask) * self.cfg.l1_weight
            total = total + loss_l1
            components["l1"] = loss_l1.item()

        if self.ssim is not None:
            loss_ssim = self.ssim(pred, target, mask) * self.cfg.ssim_weight
            total = total + loss_ssim
            components["ssim"] = loss_ssim.item()

        if self.grad is not None:
            loss_grad = self.grad(pred, target) * self.cfg.gradient_weight
            total = total + loss_grad
            components["gradient"] = loss_grad.item()

        if disc_loss is not None and self.cfg.adversarial_weight > 0:
            loss_adv = disc_loss * self.cfg.adversarial_weight
            total = total + loss_adv
            components["adversarial"] = loss_adv.item()

        components["total"] = total.item()
        return total, components
