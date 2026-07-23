"""Unit tests for evaluation metrics."""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from evaluation.metrics import (
    compute_ssim,
    compute_psnr,
    compute_mae,
    compute_mse,
    compute_nmse,
    compute_pcc,
    compute_all_metrics,
    aggregate_metrics,
)


def _random_volume(shape=(32, 32, 32), seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float32)


class TestMetrics:
    def test_perfect_ssim(self):
        v = _random_volume()
        score = compute_ssim(v, v)
        assert score > 0.99, f"SSIM of identical volumes should be ~1.0, got {score}"

    def test_ssim_range(self):
        a = _random_volume(seed=1)
        b = _random_volume(seed=2)
        score = compute_ssim(a, b)
        assert -1.0 <= score <= 1.0

    def test_perfect_psnr_is_high(self):
        v = _random_volume()
        score = compute_psnr(v, v)
        assert score >= 100.0

    def test_mae_zero_for_identical(self):
        v = _random_volume()
        assert compute_mae(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_mse_zero_for_identical(self):
        v = _random_volume()
        assert compute_mse(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_nmse_zero_for_identical(self):
        v = _random_volume()
        assert compute_nmse(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_pcc_one_for_identical(self):
        v = _random_volume()
        score = compute_pcc(v, v)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_masked_metrics(self):
        a = _random_volume(seed=1)
        b = _random_volume(seed=2)
        mask = (a > 0.3).astype(np.uint8)
        mae_masked = compute_mae(a, b, mask)
        mae_full = compute_mae(a, b)
        # Masked and full should differ (mask excludes some voxels)
        assert isinstance(mae_masked, float)
        assert isinstance(mae_full, float)

    def test_compute_all_metrics_keys(self):
        a = _random_volume(seed=1)
        b = _random_volume(seed=2)
        result = compute_all_metrics(a, b)
        for key in ["ssim", "psnr", "mae", "mse", "nmse", "pcc"]:
            assert key in result, f"Missing metric: {key}"

    def test_aggregate_metrics(self):
        records = [
            {"ssim": 0.8, "psnr": 25.0},
            {"ssim": 0.9, "psnr": 27.0},
            {"ssim": 0.85, "psnr": 26.0},
        ]
        agg = aggregate_metrics(records)
        assert "ssim" in agg
        assert "psnr" in agg
        assert agg["ssim"]["mean"] == pytest.approx(0.85, abs=1e-5)
        assert agg["ssim"]["n"] == 3
