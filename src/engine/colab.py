"""Colab/GPU environment utilities.

Handles: GPU report, disk space, Drive mount, cache check, checkpoint resume.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEPARATOR = "─" * 60


def report_hardware() -> dict:
    """Detect and report GPU, VRAM, RAM, and disk resources.

    Returns:
        Dict with hardware info fields.
    """
    info: dict = {}

    # GPU
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_free = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated(0)) / 1024**3
            info["gpu"] = gpu_name
            info["vram_total_gb"] = round(vram_total, 2)
            info["vram_free_gb"] = round(vram_free, 2)
        else:
            info["gpu"] = "NONE"
            info["vram_total_gb"] = 0
            info["vram_free_gb"] = 0
    except ImportError:
        info["gpu"] = "torch_not_installed"

    # RAM
    try:
        import psutil  # noqa: PLC0415
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / 1024**3, 2)
        info["ram_available_gb"] = round(ram.available / 1024**3, 2)
    except ImportError:
        info["ram_total_gb"] = None
        info["ram_available_gb"] = None

    # Disk (local)
    local_disk = shutil.disk_usage("/")
    info["local_disk_free_gb"] = round(local_disk.free / 1024**3, 2)
    info["local_disk_total_gb"] = round(local_disk.total / 1024**3, 2)

    lines = [
        "",
        SEPARATOR,
        "Hardware Report",
        SEPARATOR,
        f"GPU               : {info.get('gpu', 'unknown')}",
        f"VRAM total        : {info.get('vram_total_gb', 'N/A')} GB",
        f"VRAM free         : {info.get('vram_free_gb', 'N/A')} GB",
        f"RAM available     : {info.get('ram_available_gb', 'N/A')} GB",
        f"Local disk free   : {info.get('local_disk_free_gb', 'N/A')} GB",
        SEPARATOR,
        "",
    ]
    print("\n".join(lines))
    return info


def check_gpu_available() -> None:
    """Abort with a clear error if no GPU is detected.

    Raises:
        RuntimeError: If torch.cuda.is_available() returns False.
    """
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU detected. Training requires a CUDA-capable GPU. "
                "In Google Colab: Runtime → Change runtime type → GPU."
            )
    except ImportError as e:
        raise RuntimeError("PyTorch is not installed.") from e


def check_disk_space(min_gb: float, path: str = "/") -> None:
    """Warn if local disk free space is below minimum threshold.

    Args:
        min_gb: Minimum required free space in GB.
        path: Filesystem path to check (default: root).
    """
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < min_gb:
        logger.warning(
            "Low disk space: %.1f GB free (minimum required: %.1f GB). "
            "Consider freeing space before preprocessing.",
            free_gb, min_gb,
        )
    else:
        logger.info("Disk space OK: %.1f GB free.", free_gb)


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive in a Colab environment.

    Args:
        mount_point: Path where Drive should be mounted.

    Returns:
        True if mount succeeded, False if not in Colab.
    """
    try:
        from google.colab import drive  # noqa: PLC0415
        drive.mount(mount_point)
        if Path(mount_point).exists():
            logger.info("Google Drive mounted at %s.", mount_point)
            return True
        logger.error("Drive mount appeared to succeed but path %s not found.", mount_point)
        return False
    except ImportError:
        logger.info("Not in Google Colab — skipping Drive mount.")
        return False


def verify_drive_output_dir(output_root: str) -> Path:
    """Ensure the output root directory exists on Google Drive.

    Args:
        output_root: Absolute path to the output root.

    Returns:
        Resolved Path object.

    Raises:
        RuntimeError: If the parent Drive mount is not accessible.
    """
    out_path = Path(output_root)
    if not out_path.parent.parent.exists():
        raise RuntimeError(
            f"Google Drive output root parent '{out_path.parent.parent}' not accessible. "
            "Ensure Drive is mounted."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", out_path)
    return out_path


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint file by modification time.

    Args:
        checkpoint_dir: Directory containing .pt checkpoint files.

    Returns:
        Path to the latest checkpoint, or None if directory is empty.
    """
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        logger.info("Latest checkpoint found: %s", checkpoints[0])
        return checkpoints[0]
    logger.info("No checkpoints found in %s.", checkpoint_dir)
    return None


def set_reproducibility(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    """Set global random seeds and deterministic flags.

    Args:
        seed: Random seed for torch, numpy, and random.
        deterministic: If True, enforce deterministic CUDA ops.
        cudnn_benchmark: If True, enable cuDNN benchmark mode (faster but non-deterministic).
    """
    import random  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    logger.info("Reproducibility set: seed=%d, deterministic=%s, benchmark=%s.",
                seed, deterministic, cudnn_benchmark)
