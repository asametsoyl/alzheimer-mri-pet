"""Pydantic-based configuration loader with deep merge and full validation.

Usage:
    cfg = load_config("config/default.yaml", "config/experiments/my_run.yaml")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Sub-models ───────────────────────────────────────────────────────────────

class EnvironmentConfig(BaseModel):
    colab_drive_root: str = "/content/drive/MyDrive/Google Colabs"
    data_root: str = "/content/drive/MyDrive/Google Colabs/data"
    cache_root: str = "/content/cache"
    output_root: str = "/content/drive/MyDrive/Google Colabs/outputs"
    min_disk_gb: int = 10


class MRIPreprocessingConfig(BaseModel):
    target_spacing_mm: list[float] = Field(default=[1.0, 1.0, 1.0])
    skull_strip_tool: Literal["hd-bet", "synthstrip", "none"] = "hd-bet"
    bias_correction: bool = True
    mni_registration: bool = False
    normalization: Literal["z-score", "percentile", "whitestripe"] = "z-score"
    percentile_clip: list[float] = Field(default=[0.5, 99.5])
    target_shape: list[int] = Field(default=[128, 128, 128])

    @field_validator("target_spacing_mm")
    @classmethod
    def validate_spacing(cls, v: list[float]) -> list[float]:
        """Ensure spacing has exactly 3 positive values."""
        if len(v) != 3 or any(s <= 0 for s in v):
            raise ValueError("target_spacing_mm must have 3 positive floats.")
        return v

    @field_validator("target_shape")
    @classmethod
    def validate_shape(cls, v: list[int]) -> list[int]:
        """Ensure shape has exactly 3 positive values divisible by 16."""
        if len(v) != 3:
            raise ValueError("target_shape must have 3 integers.")
        for s in v:
            if s <= 0 or s % 16 != 0:
                raise ValueError(f"Each target_shape dimension must be positive and divisible by 16, got {s}.")
        return v


class PETPreprocessingConfig(BaseModel):
    coregister_to_mri: bool = True
    amyloid_suvr_threshold: dict[str, float] = Field(
        default={"Florbetapir": 1.11, "Florbetaben": 1.08}
    )
    reference_region: dict[str, str] = Field(
        default={
            "FDG": "whole_cerebellum",
            "Florbetapir": "cerebellar_gray",
            "Florbetaben": "cerebellar_gray",
            "Flortaucipir": "inferior_cerebellar_gray",
        }
    )


class PreprocessingConfig(BaseModel):
    mri: MRIPreprocessingConfig = Field(default_factory=MRIPreprocessingConfig)
    pet: PETPreprocessingConfig = Field(default_factory=PETPreprocessingConfig)


class DataConfig(BaseModel):
    dataset: Literal["ADNI", "AIBL", "OASIS", "custom"] = "ADNI"
    mri_modality: str = "T1w"
    pet_tracer: str = "auto"
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    exclusions_file: str = "data/exclusions.csv"

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DataConfig":
        """Ensure split ratios sum to 1.0."""
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
        return self


class PatchConfig(BaseModel):
    size: list[int] = Field(default=[96, 96, 96])
    stride_train: list[int] = Field(default=[48, 48, 48])
    stride_inference: list[int] = Field(default=[64, 64, 64])
    sampling_strategy: Literal["weighted_random", "uniform", "foreground_weighted"] = "weighted_random"
    foreground_threshold: float = 0.05
    min_brain_fraction: float = 0.10

    @field_validator("size")
    @classmethod
    def validate_patch_size(cls, v: list[int]) -> list[int]:
        """Patch size must be divisible by 16."""
        if len(v) != 3:
            raise ValueError("Patch size must have 3 integers.")
        for s in v:
            if s % 16 != 0:
                raise ValueError(f"Patch dimension {s} must be divisible by 16.")
        return v


class AugmentationConfig(BaseModel):
    enabled: bool = True
    flip_lr_prob: float = 0.5
    affine_prob: float = 0.3
    affine_degrees: float = 10.0
    affine_scale: list[float] = Field(default=[0.9, 1.1])
    intensity_shift_prob: float = 0.2
    intensity_shift_fraction: float = 0.05
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.01
    bias_field_prob: float = 0.10
    elastic_prob: float = 0.10


class ModelConfig(BaseModel):
    name: Literal["res_attn_unet", "attn_unet", "unetr", "swin_unetr", "mednext"] = "res_attn_unet"
    in_channels: int = 1
    out_channels: int = 1
    features: list[int] = Field(default=[32, 64, 128, 256, 512])
    dropout: float = 0.1
    deep_supervision: bool = False


class LossConfig(BaseModel):
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    perceptual_weight: float = 0.0
    adversarial_weight: float = 0.0
    gradient_weight: float = 0.0

    @model_validator(mode="after")
    def validate_positive_weights(self) -> "LossConfig":
        """All weights must be non-negative and at least one must be positive."""
        weights = [
            self.l1_weight, self.ssim_weight, self.perceptual_weight,
            self.adversarial_weight, self.gradient_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All loss weights must be >= 0.")
        if sum(weights) == 0:
            raise ValueError("At least one loss weight must be > 0.")
        return self


class TrainingConfig(BaseModel):
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine_annealing_warm_restarts"
    lr_T0: int = 10
    lr_eta_min: float = 1e-6
    max_epochs: int = 200
    early_stopping_patience: int = 30
    early_stopping_metric: str = "val_ssim"
    gradient_clip_norm: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    save_every_n_epochs: int = 10
    save_best_metric: str = "val_ssim"
    amp_enabled: bool = True


class InferenceConfig(BaseModel):
    overlap: float = 0.5
    blend_mode: Literal["gaussian", "constant"] = "gaussian"
    tta_enabled: bool = False
    tta_flips: list[str] = Field(default=["lr"])
    save_difference_map: bool = True
    save_subject_metrics: bool = True


class EvaluationConfig(BaseModel):
    compute_on_brain_mask: bool = True
    metrics: list[str] = Field(default=["ssim", "psnr", "mae", "mse", "nmse", "pcc"])
    clinical_metrics: bool = True
    roi_labels: list[str] = Field(
        default=["frontal", "temporal", "parietal", "cingulate", "precuneus", "striatum"]
    )
    statistical_test: Literal["wilcoxon", "ttest"] = "wilcoxon"
    multiple_comparison_correction: Literal["bonferroni", "fdr", "none"] = "bonferroni"


class TrackingConfig(BaseModel):
    wandb_enabled: bool = False
    wandb_project: str = "mri-pet-synthesis"
    log_every_n_steps: int = 10
    metrics_file: str = "metrics.jsonl"
    experiments_index: str = "experiments.csv"


class ReproducibilityConfig(BaseModel):
    random_seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False


# ─── Root Config ──────────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """Root configuration for the MRI→PET synthesis pipeline."""

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


# ─── Loader ───────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Args:
        base: The default configuration dictionary.
        override: The experiment-specific overrides.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    default_path: str | Path,
    experiment_path: Optional[str | Path] = None,
) -> PipelineConfig:
    """Load and validate pipeline configuration.

    Loads the default config, optionally deep-merges an experiment override,
    then validates the merged result using Pydantic.

    Args:
        default_path: Path to config/default.yaml.
        experiment_path: Optional path to config/experiments/<name>.yaml.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If default_path does not exist.
        pydantic.ValidationError: If any field fails validation.
    """
    default_path = Path(default_path)
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with open(default_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if experiment_path is not None:
        exp_path = Path(experiment_path)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        with open(exp_path, encoding="utf-8") as f:
            exp_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, exp_raw)

    return PipelineConfig(**raw)


def config_to_dict(cfg: PipelineConfig) -> dict:
    """Serialize config to a plain dictionary (for YAML/JSON saving).

    Args:
        cfg: Validated PipelineConfig.

    Returns:
        Serializable dictionary representation.
    """
    return cfg.model_dump()
