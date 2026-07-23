"""Full training loop with AMP, gradient accumulation, checkpointing, and early stopping.

NaN detection → debug checkpoint + exception.
OOM → informative error with config suggestions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import PipelineConfig, TrainingConfig
from losses.combined import CombinedLoss
from models.base import BaseModel

logger = logging.getLogger(__name__)


# ─── Optimizer factory ────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from training config.

    Args:
        model: Model whose parameters will be optimized.
        cfg: Training configuration.

    Returns:
        Configured optimizer.
    """
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: '{cfg.optimizer}'")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig):
    """Build LR scheduler from training config.

    Args:
        optimizer: Optimizer to schedule.
        cfg: Training configuration.

    Returns:
        LR scheduler or None.
    """
    name = cfg.lr_scheduler
    if name == "cosine_annealing_warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.lr_T0, eta_min=cfg.lr_eta_min)
    if name == "step_lr":
        return StepLR(optimizer, step_size=cfg.lr_T0, gamma=0.1)
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=10, min_lr=cfg.lr_eta_min)
    if name == "none":
        return None
    logger.warning("Unknown scheduler '%s' — no scheduler will be used.", name)
    return None


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def _ssim_score(pred: Tensor, target: Tensor) -> float:
    """Compute mean SSIM (simplified for training monitoring, not full 3D).

    Args:
        pred: Predicted tensor.
        target: Target tensor.

    Returns:
        SSIM value as float.
    """
    from evaluation.metrics import compute_ssim  # noqa: PLC0415
    return float(compute_ssim(pred.detach().cpu().numpy(), target.detach().cpu().numpy()))


def _psnr_score(pred: Tensor, target: Tensor) -> float:
    """Compute PSNR for training monitoring.

    Args:
        pred: Predicted tensor.
        target: Target tensor.

    Returns:
        PSNR value in dB.
    """
    mse = float(torch.mean((pred - target) ** 2).item())
    if mse < 1e-10:
        return 100.0
    data_range = float(target.max() - target.min())
    return float(20 * torch.log10(torch.tensor(data_range / (mse ** 0.5 + 1e-8))).item())


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """Full training loop for MRI→PET synthesis.

    Features:
    - AMP (Automatic Mixed Precision)
    - Gradient accumulation
    - Gradient clipping
    - Early stopping
    - Best model + periodic checkpoint saving
    - NaN detection → debug checkpoint + exception
    - OOM → informative error with suggestions
    - Per-epoch structured logging

    Args:
        model: The synthesis model.
        loss_fn: CombinedLoss instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Full pipeline configuration.
        run_dir: Directory to save checkpoints and logs for this run.
    """

    def __init__(
        self,
        model: BaseModel,
        loss_fn: CombinedLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: PipelineConfig,
        run_dir: Path,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.t_cfg = cfg.training
        self.run_dir = run_dir

        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = build_optimizer(self.model, self.t_cfg)
        self.scheduler = build_scheduler(self.optimizer, self.t_cfg)
        self.scaler = GradScaler(enabled=self.t_cfg.amp_enabled)

        self.best_val_metric: float = -float("inf")
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.global_step: int = 0
        self.start_epoch: int = 0

        self.metrics_path = self.log_dir / "metrics.jsonl"

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, tag: str = "periodic") -> Path:
        """Save a full training checkpoint.

        Args:
            epoch: Current epoch number.
            tag: Label for the checkpoint filename ('best' or 'periodic').

        Returns:
            Path to the saved checkpoint file.
        """
        path = self.checkpoint_dir / f"checkpoint_{tag}_epoch{epoch:04d}.pt"
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            "config": self.cfg.model_dump(),
        }
        torch.save(state, str(path))
        logger.info("Checkpoint saved: %s", path.name)
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.

        Returns:
            Epoch to resume from.
        """
        state = torch.load(str(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.best_val_metric = state.get("best_val_metric", -float("inf"))
        self.global_step = state.get("global_step", 0)
        resume_epoch = state["epoch"] + 1
        logger.info("Resumed from checkpoint: %s (epoch %d).", checkpoint_path.name, resume_epoch)
        return resume_epoch

    # ── Train one epoch ───────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of training metrics (loss components).
        """
        self.model.train()
        accum_steps = self.t_cfg.gradient_accumulation_steps
        self.optimizer.zero_grad()

        running: dict[str, float] = {}
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            mri = batch["mri"].to(self.device, non_blocking=True)
            pet = batch["pet"].to(self.device, non_blocking=True)

            try:
                with torch.cuda.amp.autocast(enabled=self.t_cfg.amp_enabled):
                    pred = self.model(mri)
                    loss, components = self.loss_fn(pred, pet)
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "VRAM Out of Memory. Suggestions:\n"
                        "  1. Reduce training.batch_size\n"
                        "  2. Reduce patch.size\n"
                        "  3. Enable gradient checkpointing\n"
                        f"  Current batch_size={self.t_cfg.batch_size}, patch_size={self.cfg.patch.size}"
                    ) from e
                raise

            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                debug_path = self.checkpoint_dir / f"debug_nan_epoch{epoch}_batch{batch_idx}.pt"
                torch.save({"model": self.model.state_dict(), "batch": batch}, str(debug_path))
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. "
                    f"Debug checkpoint saved to {debug_path}."
                )

            # Accumulate metrics
            for k, v in components.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            # Gradient step every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.t_cfg.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

        # Normalize
        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ── Validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation loop.

        Returns:
            Dict of validation metrics.
        """
        self.model.eval()
        running: dict[str, float] = {}
        n_batches = 0
        ssim_vals: list[float] = []
        psnr_vals: list[float] = []

        for batch in self.val_loader:
            mri = batch["mri"].to(self.device, non_blocking=True)
            pet = batch["pet"].to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.t_cfg.amp_enabled):
                pred = self.model(mri)
                _, components = self.loss_fn(pred, pet)

            for k, v in components.items():
                running[k] = running.get(k, 0.0) + v
            ssim_vals.append(_ssim_score(pred, pet))
            psnr_vals.append(_psnr_score(pred, pet))
            n_batches += 1

        result = {f"val_{k}": v / max(n_batches, 1) for k, v in running.items()}
        result["val_ssim"] = sum(ssim_vals) / max(len(ssim_vals), 1)
        result["val_psnr"] = sum(psnr_vals) / max(len(psnr_vals), 1)
        return result

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self, resume_checkpoint: Optional[Path] = None) -> None:
        """Run the full training loop.

        Args:
            resume_checkpoint: Optional path to resume from a saved checkpoint.
        """
        if resume_checkpoint is not None:
            self.start_epoch = self.load_checkpoint(resume_checkpoint)

        logger.info(
            "Training started | device=%s | epochs=%d | AMP=%s | accum=%d",
            self.device, self.t_cfg.max_epochs,
            self.t_cfg.amp_enabled, self.t_cfg.gradient_accumulation_steps,
        )

        for epoch in range(self.start_epoch, self.t_cfg.max_epochs):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]
            vram_gb = (torch.cuda.memory_allocated(0) / 1024**3
                       if torch.cuda.is_available() else 0.0)
            logger.info(
                "[Epoch %d/%d] LR=%.2e | VRAM=%.2f GB",
                epoch + 1, self.t_cfg.max_epochs, current_lr, vram_gb,
            )

            # Train
            train_metrics = self._train_epoch(epoch)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    pass  # stepped after validation below
                else:
                    self.scheduler.step()

            # Validate
            val_metrics = self._validate()

            # ReduceLROnPlateau needs val metric
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get("val_ssim", 0.0))

            elapsed = time.time() - epoch_start

            # Logging
            log_line = (
                f"[Epoch {epoch+1}/{self.t_cfg.max_epochs}] "
                f"train_loss={train_metrics.get('total', 0):.4f} | "
                f"val_loss={val_metrics.get('val_total', 0):.4f} | "
                f"val_ssim={val_metrics.get('val_ssim', 0):.4f} | "
                f"val_psnr={val_metrics.get('val_psnr', 0):.2f} dB | "
                f"time={elapsed:.1f}s"
            )
            logger.info(log_line)

            # Write metrics to JSONL
            epoch_record = {"epoch": epoch + 1, **train_metrics, **val_metrics, "elapsed_s": elapsed}
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(epoch_record) + "\n")

            # Best model check
            monitor = val_metrics.get(self.t_cfg.save_best_metric, val_metrics.get("val_ssim", 0))
            if monitor > self.best_val_metric:
                self.best_val_metric = monitor
                self.best_val_loss = val_metrics.get("val_total", float("inf"))
                self.epochs_without_improvement = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), str(best_path))
                logger.info("New best model saved (val_%s=%.4f).",
                            self.t_cfg.save_best_metric, monitor)
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.t_cfg.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1, tag="periodic")

            # Early stopping
            if self.epochs_without_improvement >= self.t_cfg.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.t_cfg.early_stopping_patience,
                )
                break

        self.save_checkpoint(epoch + 1, tag="final")
        logger.info("Training complete. Best val_%s=%.4f.",
                    self.t_cfg.save_best_metric, self.best_val_metric)
