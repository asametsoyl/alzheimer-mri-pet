"""Unit tests for config loading, validation, and deep merge."""

import pytest
from pydantic import ValidationError

import sys
sys.path.insert(0, "src")

from config import load_config, PipelineConfig, DataConfig, PatchConfig, LossConfig
from pathlib import Path


CONFIG_PATH = Path("config/default.yaml")


class TestConfigLoad:
    def test_load_default_config(self):
        """Default config loads and validates without error."""
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg, PipelineConfig)

    def test_default_dataset(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg.data.dataset == "ADNI"

    def test_default_model(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg.model.name == "res_attn_unet"

    def test_default_loss_weights_positive(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg.loss.l1_weight >= 0
        assert cfg.loss.ssim_weight >= 0

    def test_split_ratios_sum_to_one(self):
        cfg = load_config(CONFIG_PATH)
        total = round(cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio, 6)
        assert abs(total - 1.0) < 1e-5

    def test_target_shape_divisible_by_16(self):
        cfg = load_config(CONFIG_PATH)
        for s in cfg.preprocessing.mri.target_shape:
            assert s % 16 == 0, f"Shape dim {s} not divisible by 16"

    def test_patch_size_divisible_by_16(self):
        cfg = load_config(CONFIG_PATH)
        for s in cfg.patch.size:
            assert s % 16 == 0


class TestConfigValidation:
    def test_invalid_split_ratios_raises(self):
        with pytest.raises(ValidationError):
            DataConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValidationError):
            PatchConfig(size=[100, 100, 100])  # not divisible by 16

    def test_all_zero_loss_weights_raises(self):
        with pytest.raises(ValidationError):
            LossConfig(l1_weight=0.0, ssim_weight=0.0)

    def test_negative_loss_weight_raises(self):
        with pytest.raises(ValidationError):
            LossConfig(l1_weight=-1.0)

    def test_missing_default_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")
