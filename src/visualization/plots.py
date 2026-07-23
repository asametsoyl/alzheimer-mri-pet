"""Visualization: training curves, axial overlays, Bland-Altman plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(
    metrics_jsonl_path: Path,
    output_dir: Path,
) -> None:
    """Generate training loss, SSIM, and PSNR curves from the metrics JSONL log.

    Args:
        metrics_jsonl_path: Path to the metrics.jsonl file.
        output_dir: Directory to save PNG plots.
    """
    import json  # noqa: PLC0415

    records = []
    with open(metrics_jsonl_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    epochs = [r["epoch"] for r in records]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [r.get("total", 0) for r in records], label="Train Loss", color="#4C72B0")
    ax.plot(epochs, [r.get("val_total", 0) for r in records], label="Val Loss",
            color="#DD8452", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_dir / "loss_curve.png"), dpi=150)
    plt.close(fig)

    # SSIM curve
    if "val_ssim" in records[0]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, [r.get("val_ssim", 0) for r in records], color="#55A868")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation SSIM")
        ax.set_title("Validation SSIM over Epochs")
        fig.tight_layout()
        fig.savefig(str(output_dir / "ssim_curve.png"), dpi=150)
        plt.close(fig)

    # PSNR curve
    if "val_psnr" in records[0]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, [r.get("val_psnr", 0) for r in records], color="#C44E52")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation PSNR (dB)")
        ax.set_title("Validation PSNR over Epochs")
        fig.tight_layout()
        fig.savefig(str(output_dir / "psnr_curve.png"), dpi=150)
        plt.close(fig)

    logger.info("Training curves saved to %s.", output_dir)


# ─── Axial Overlays ───────────────────────────────────────────────────────────

def plot_overlay(
    mri: np.ndarray,
    real_pet: np.ndarray,
    synth_pet: np.ndarray,
    output_path: Path,
    subject_id: str = "",
    slice_idx: Optional[int] = None,
) -> None:
    """Save a 4-panel axial overlay: MRI | Real PET | Synth PET | Difference.

    Args:
        mri: 3D MRI array (D, H, W).
        real_pet: 3D real PET array.
        synth_pet: 3D synthesized PET array.
        output_path: Path to save the figure (.png).
        subject_id: For figure title.
        slice_idx: Axial slice index. Defaults to middle slice.
    """
    if slice_idx is None:
        slice_idx = mri.shape[0] // 2

    diff = synth_pet - real_pet
    abs_diff = np.abs(diff)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        (mri[slice_idx], "MRI", "gray"),
        (real_pet[slice_idx], "Real PET", "hot"),
        (synth_pet[slice_idx], "Synth PET", "hot"),
        (abs_diff[slice_idx], "|Difference|", "RdBu_r"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        im = ax.imshow(img.T, cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Subject: {subject_id} | Axial slice {slice_idx}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Overlay saved: %s", output_path)


# ─── Bland-Altman Plot ────────────────────────────────────────────────────────

def plot_bland_altman(
    real_suvr: list[float],
    synth_suvr: list[float],
    output_path: Path,
    title: str = "Bland-Altman: Real vs Synthesized PET SUVr",
) -> dict:
    """Generate a Bland-Altman agreement plot for SUVr values.

    Args:
        real_suvr: List of real PET SUVr values (one per subject).
        synth_suvr: List of synthesized PET SUVr values (one per subject).
        output_path: Path to save the figure.
        title: Plot title.

    Returns:
        Dict with bias, LoA_lower, LoA_upper.
    """
    real = np.array(real_suvr)
    synth = np.array(synth_suvr)

    mean_val = (real + synth) / 2
    diff = synth - real
    bias = float(np.mean(diff))
    std_diff = float(np.std(diff))
    loa_lower = bias - 1.96 * std_diff
    loa_upper = bias + 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_val, diff, alpha=0.7, color="#4C72B0", edgecolors="white", s=60)
    ax.axhline(bias, color="#DD8452", linewidth=2, label=f"Bias: {bias:.3f}")
    ax.axhline(loa_upper, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"+1.96 SD: {loa_upper:.3f}")
    ax.axhline(loa_lower, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"−1.96 SD: {loa_lower:.3f}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Mean of Real and Synthesized SUVr", fontsize=12)
    ax.set_ylabel("Synthesized − Real SUVr", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Bland-Altman: bias=%.3f, LoA=[%.3f, %.3f]", bias, loa_lower, loa_upper)
    return {"bias": bias, "loa_lower": loa_lower, "loa_upper": loa_upper, "std_diff": std_diff}
