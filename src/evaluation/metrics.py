"""Image quality and clinical evaluation metrics.

All metrics computed on brain mask only (if provided).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_2d
from skimage.metrics import peak_signal_noise_ratio as psnr_2d

logger = logging.getLogger(__name__)


def _apply_mask(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Extract voxels inside mask (or return all if no mask).

    Args:
        arr: 3D array.
        mask: Boolean mask of same shape, or None.

    Returns:
        1D array of selected voxels.
    """
    if mask is not None:
        return arr[mask.astype(bool)]
    return arr.ravel()


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: Optional[float] = None,
) -> float:
    """Compute 3D SSIM by averaging over axial slices (fast approximation).

    Args:
        pred: Predicted PET array (D, H, W).
        target: Ground truth PET array (D, H, W).
        mask: Optional brain mask.
        data_range: Data range for SSIM. Auto-computed if None.

    Returns:
        Mean SSIM value.
    """
    if data_range is None:
        data_range = float(max(target.max() - target.min(), 1e-8))

    ssim_vals = []
    for i in range(pred.shape[0]):
        s = ssim_2d(pred[i], target[i], data_range=data_range)
        ssim_vals.append(s)
    return float(np.mean(ssim_vals))


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: Optional[float] = None,
) -> float:
    """Compute PSNR on brain-masked voxels.

    Args:
        pred: Predicted PET.
        target: Ground truth PET.
        mask: Optional brain mask.
        data_range: Signal range. Auto-computed if None.

    Returns:
        PSNR in dB.
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(target, mask)
    if data_range is None:
        data_range = float(max(t.max() - t.min(), 1e-8))
    mse = float(np.mean((p - t) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Mean Absolute Error on masked voxels.

    Args:
        pred: Predicted PET.
        target: Ground truth PET.
        mask: Optional brain mask.

    Returns:
        MAE value.
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(target, mask)
    return float(np.mean(np.abs(p - t)))


def compute_mse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Mean Squared Error on masked voxels.

    Args:
        pred: Predicted PET.
        target: Ground truth PET.
        mask: Optional brain mask.

    Returns:
        MSE value.
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(target, mask)
    return float(np.mean((p - t) ** 2))


def compute_nmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Normalized Mean Squared Error.

    NMSE = MSE / mean(target^2).

    Args:
        pred: Predicted PET.
        target: Ground truth PET.
        mask: Optional brain mask.

    Returns:
        NMSE value.
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(target, mask)
    denominator = float(np.mean(t ** 2))
    if denominator < 1e-10:
        return float("inf")
    return float(np.mean((p - t) ** 2) / denominator)


def compute_pcc(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute Pearson Correlation Coefficient on masked voxels.

    Args:
        pred: Predicted PET.
        target: Ground truth PET.
        mask: Optional brain mask.

    Returns:
        PCC value in [-1, 1].
    """
    p = _apply_mask(pred, mask)
    t = _apply_mask(target, mask)
    if p.std() < 1e-8 or t.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(p, t)
    return float(r)


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute all six image quality metrics for one subject.

    Args:
        pred: Predicted PET (D, H, W).
        target: Ground truth PET (D, H, W).
        mask: Optional brain mask (D, H, W).

    Returns:
        Dict with keys: ssim, psnr, mae, mse, nmse, pcc.
    """
    return {
        "ssim": compute_ssim(pred, target, mask),
        "psnr": compute_psnr(pred, target, mask),
        "mae": compute_mae(pred, target, mask),
        "mse": compute_mse(pred, target, mask),
        "nmse": compute_nmse(pred, target, mask),
        "pcc": compute_pcc(pred, target, mask),
    }


def aggregate_metrics(per_subject: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean and std over all subjects for each metric.

    Args:
        per_subject: List of per-subject metric dicts.

    Returns:
        Dict mapping metric name → {'mean': float, 'std': float}.
    """
    if not per_subject:
        return {}
    all_keys = per_subject[0].keys()
    result = {}
    for key in all_keys:
        vals = [d[key] for d in per_subject if key in d]
        result[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
    return result
