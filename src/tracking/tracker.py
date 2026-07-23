"""Experiment tracker: run_id generation, metrics JSONL, optional W&B, experiments.csv."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks experiment metadata, metrics, and optional W&B integration.

    Args:
        run_dir: Directory for this run's outputs.
        config_dict: Full pipeline config as a serializable dict.
        wandb_enabled: Whether to try W&B logging.
        wandb_project: W&B project name.
        experiments_index: Path to the global experiments.csv.
    """

    def __init__(
        self,
        run_dir: Path,
        config_dict: dict,
        wandb_enabled: bool = False,
        wandb_project: str = "mri-pet-synthesis",
        experiments_index: Optional[Path] = None,
    ) -> None:
        self.run_id = self._generate_run_id()
        self.run_dir = run_dir
        self.config_dict = config_dict
        self.experiments_index = experiments_index
        self._wandb = None

        if wandb_enabled:
            self._init_wandb(wandb_project)

        logger.info("ExperimentTracker initialized | run_id=%s", self.run_id)

    def _generate_run_id(self) -> str:
        """Generate a unique run ID using timestamp + hash.

        Returns:
            String like '20240115_143022_a3f9'.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h = hashlib.sha1(ts.encode()).hexdigest()[:4]
        return f"{ts}_{h}"

    def _init_wandb(self, project: str) -> None:
        """Initialize Weights & Biases if API key is available.

        Args:
            project: W&B project name.
        """
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logger.info("W&B: WANDB_API_KEY not set — skipping W&B logging.")
            return
        try:
            import wandb  # noqa: PLC0415
            wandb.init(project=project, id=self.run_id, config=self.config_dict)
            self._wandb = wandb
            logger.info("W&B initialized: project=%s, run_id=%s.", project, self.run_id)
        except ImportError:
            logger.warning("W&B not installed — skipping. Install with: pip install wandb")

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a metrics dict at a given step.

        Args:
            metrics: Dict of metric names to float values.
            step: Optional global step or epoch number.
        """
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_config(self, output_path: Path) -> None:
        """Save the full run config as run_config.yaml.

        Args:
            output_path: Path to save the YAML file.
        """
        import yaml  # noqa: PLC0415
        payload = {"run_id": self.run_id, **self.config_dict}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
        logger.info("Run config saved: %s", output_path)

    def log_final_metrics(self, final_metrics: dict) -> None:
        """Append this run's final metrics to the global experiments.csv.

        Args:
            final_metrics: Dict of final evaluation metrics.
        """
        if self.experiments_index is None:
            return

        row = {"run_id": self.run_id, **final_metrics}
        file_exists = self.experiments_index.exists()
        with open(self.experiments_index, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Appended final metrics to %s.", self.experiments_index)

    def finish(self) -> None:
        """Finalize the experiment tracker (close W&B run if active)."""
        if self._wandb is not None:
            self._wandb.finish()
            logger.info("W&B run finished.")
