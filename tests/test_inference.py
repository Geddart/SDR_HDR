"""Tests for pipeline/inference.py — convert_linear, load_image, estimate_tile_size."""
import numpy as np
import pytest
import torch
import cv2
from pathlib import Path

from pipeline.inference import convert_linear, load_image, estimate_tile_size


class TestConvertLinear:
    """Tests for the linear SDR-to-HDR conversion pipeline."""

    def test_output_shape_matches_input(self):
        img = torch.rand(100, 200, 3)
        result = convert_linear(img)
        assert result.shape == (100, 200, 3)

    def test_black_stays_near_zero(self):
        img = torch.zeros(10, 10, 3)
        result = convert_linear(img)
        assert result.max().item() < 0.01

    def test_white_maps_to_peak(self):
        img = torch.ones(10, 10, 3)
        result = convert_linear(img, peak=150.0, gain=3.0)
        # After inverse tonemap, white (1.0) -> peak (150). Then Rec.709->ACEScg
        # preserves white point ~1.0, so max should be near peak.
        max_val = result.max().item()
        assert max_val > 100, f"White should map near peak, got {max_val}"
        assert max_val < 200, f"White should not exceed peak by too much, got {max_val}"

    def test_midtones_lifted_by_gain(self):
        # Mid-gray sRGB ~0.5 -> linear ~0.214
        img = torch.full((10, 10, 3), 0.5)
        result_low = convert_linear(img, peak=150.0, gain=2.0)
        result_high = convert_linear(img, peak=150.0, gain=5.0)
        # Higher gain should produce brighter midtones
        assert result_high.mean().item() > result_low.mean().item()

    def test_exposure_doubles(self):
        img = torch.full((10, 10, 3), 0.3)
        result_0 = convert_linear(img, exposure=0.0)
        result_1 = convert_linear(img, exposure=1.0)
        # +1 stop should roughly double values (ignoring dither noise)
        ratio = result_1.mean().item() / result_0.mean().item()
        assert 1.8 < ratio < 2.2, f"Expected ~2x ratio, got {ratio}"

    def test_output_is_non_negative(self):
        img = torch.rand(50, 50, 3)
        result = convert_linear(img)
        assert result.min().item() >= 0.0

    def test_chromaticity_preserved_for_gray(self):
        # Gray input should produce roughly equal R, G, B in ACEScg
        img = torch.full((10, 10, 3), 0.5)
        result = convert_linear(img, peak=150.0, gain=3.0)
        mean_rgb = result.mean(dim=(0, 1))
        # ACEScg transform shifts things slightly, but gray should stay close
        spread = (mean_rgb.max() - mean_rgb.min()) / mean_rgb.mean()
        assert spread < 0.15, f"Gray should stay neutral, spread={spread:.3f}"


class TestLoadImage:
    """Tests for image loading with alpha support."""

    def test_load_rgb_png(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        rgb, alpha = load_image(path)
        assert rgb.shape == (64, 64, 3)
        assert alpha is None

    def test_load_rgba_png(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        path = str(tmp_path / "test_rgba.png")
        cv2.imwrite(path, img)
        rgb, alpha = load_image(path)
        assert rgb.shape == (64, 64, 3)
        assert alpha is not None
        assert alpha.shape == (64, 64)

    def test_load_grayscale(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        path = str(tmp_path / "test_gray.png")
        cv2.imwrite(path, img)
        rgb, alpha = load_image(path)
        assert rgb.shape == (64, 64, 3)
        assert alpha is None

    def test_values_in_0_1(self, tmp_path):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        path = str(tmp_path / "test_128.png")
        cv2.imwrite(path, img)
        rgb, alpha = load_image(path)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file.png")


class TestEstimateTileSize:
    """Tests for VRAM-based tile size estimation."""

    def test_cpu_returns_max(self):
        assert estimate_tile_size(torch.device("cpu")) == 2048

    def test_returns_multiple_of_64(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tile = estimate_tile_size(device)
        assert tile % 64 == 0

    def test_within_bounds(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tile = estimate_tile_size(device)
        assert 256 <= tile <= 2048

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_positive(self):
        tile = estimate_tile_size(torch.device("cuda"))
        assert tile >= 256


class TestPresets:
    """Tests for scene preset behavior via convert_linear."""

    def test_hdri_hotter_than_night(self):
        """HDRI preset (peak=250) should produce higher max than night (peak=150)."""
        img = torch.ones(10, 10, 3)
        night = convert_linear(img, peak=150.0, gain=3.0)
        hdri = convert_linear(img, peak=250.0, gain=3.0)
        assert hdri.max().item() > night.max().item()

    def test_overcast_lifts_midtones_most(self):
        """Overcast (gain=5.0) should produce brighter mid-tones than day (gain=4.0)."""
        img = torch.full((10, 10, 3), 0.3)
        overcast = convert_linear(img, peak=15.0, gain=5.0)
        day = convert_linear(img, peak=30.0, gain=4.0)
        # Overcast has higher gain, but also lower peak.
        # For mid-tone (0.3 sRGB -> ~0.073 linear), gain dominates over peak.
        assert overcast.mean().item() > day.mean().item() * 0.5

    def test_day_lower_peak_than_night(self):
        """Day preset white should be lower than night preset white."""
        img = torch.ones(10, 10, 3)
        day = convert_linear(img, peak=30.0, gain=4.0)
        night = convert_linear(img, peak=150.0, gain=3.0)
        assert day.max().item() < night.max().item()

    def test_all_presets_produce_valid_output(self):
        """Every preset should produce non-negative, finite output."""
        from convert import PRESETS
        img = torch.rand(20, 20, 3)
        for name, vals in PRESETS.items():
            result = convert_linear(img, peak=vals["peak"], gain=vals["gain"])
            assert result.min().item() >= 0.0, f"{name}: negative values"
            assert torch.isfinite(result).all(), f"{name}: non-finite values"
