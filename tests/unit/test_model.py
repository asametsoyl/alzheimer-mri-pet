"""Unit tests for model forward pass, shape, NaN/Inf safety."""

import sys
sys.path.insert(0, "src")

import pytest
import torch
from models.res_attn_unet import ResAttnUNet3D
from models.factory import build_model
from config import ModelConfig


class TestResAttnUNet:
    def _make_model(self, features=None):
        return ResAttnUNet3D(
            in_channels=1,
            out_channels=1,
            features=features or [8, 16, 32, 64, 128],
            dropout=0.0,
        )

    def test_output_shape(self):
        """Output must match input spatial shape."""
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_no_nan_in_output(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "NaN in model output"
        assert not torch.isinf(out).any(), "Inf in model output"

    def test_num_parameters_positive(self):
        model = self._make_model()
        n = model.get_num_parameters()
        assert n > 0

    def test_get_config(self):
        model = self._make_model()
        cfg = model.get_config()
        assert "name" in cfg
        assert cfg["name"] == "res_attn_unet"
        assert "num_parameters" in cfg

    def test_batch_size_2(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(2, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 32, 32, 32)


class TestModelFactory:
    def test_build_res_attn_unet(self):
        cfg = ModelConfig(name="res_attn_unet", features=[8, 16, 32, 64, 128])
        model = build_model(cfg)
        assert isinstance(model, ResAttnUNet3D)

    def test_unknown_model_raises(self):
        cfg = ModelConfig(name="res_attn_unet")  # use valid name first
        cfg_dict = cfg.model_dump()
        cfg_dict["name"] = "nonexistent_model"
        from config import ModelConfig as MC  # noqa: PLC0415
        # Override with invalid registry lookup
        with pytest.raises(ValueError, match="Unknown model"):
            from models import factory  # noqa: PLC0415
            factory.build_model(type("FakeCfg", (), {"name": "nonexistent_model", "in_channels": 1,
                                                      "out_channels": 1, "features": [8], "dropout": 0.0})())
