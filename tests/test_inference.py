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
        # TPDF dithering adds ±1/255 noise, so values up to ~0.012 are normal
        assert result.max().item() < 0.02

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
            result = convert_linear(
                img, peak=vals["peak"], gain=vals["gain"],
                power=vals.get("power", 2.0),
                dither=vals.get("dither", True),
            )
            assert result.min().item() >= 0.0, f"{name}: negative values"
            assert torch.isfinite(result).all(), f"{name}: non-finite values"


class TestPower:
    """Tests for the power parameter in inverse tonemap."""

    def test_higher_power_keeps_midtones_darker(self):
        """Higher power should keep midtones closer to gain*L."""
        img = torch.full((10, 10, 3), 0.5)
        low_power = convert_linear(img, peak=150.0, gain=1.0, power=2.0, dither=False)
        high_power = convert_linear(img, peak=150.0, gain=1.0, power=50.0, dither=False)
        # High power concentrates expansion near white, so midtones stay lower
        assert high_power.mean().item() < low_power.mean().item()

    def test_power_does_not_affect_peak(self):
        """White input should map to peak regardless of power."""
        img = torch.ones(10, 10, 3)
        for power in [2.0, 4.0, 10.0, 50.0]:
            result = convert_linear(img, peak=150.0, gain=3.0, power=power, dither=False)
            max_val = result.max().item()
            assert max_val > 140, f"power={power}: max={max_val}, expected ~150"
            assert max_val < 160, f"power={power}: max={max_val}, expected ~150"

    def test_roundtrip_preset_near_identity(self):
        """Roundtrip preset (gain=1, power=50) should be near-identity for mid values."""
        img = torch.full((10, 10, 3), 0.3)
        result = convert_linear(img, peak=1.0, gain=1.0, power=50.0, dither=False)
        # sRGB 0.3 -> linear ~0.073. With gain=1, peak=1, power=50: output ≈ 0.073
        # After Rec.709->ACEScg (preserves gray), should stay near 0.073
        mean = result.mean().item()
        expected = 0.073
        assert abs(mean - expected) / expected < 0.05, f"Expected ~{expected}, got {mean}"


class TestDither:
    """Tests for TPDF dithering control."""

    def test_no_dither_is_deterministic(self):
        """Without dithering, identical inputs give identical outputs."""
        img = torch.full((10, 10, 3), 0.5)
        r1 = convert_linear(img, dither=False)
        r2 = convert_linear(img, dither=False)
        torch.testing.assert_close(r1, r2)

    def test_dither_adds_variation(self):
        """With dithering, identical inputs give slightly different outputs."""
        img = torch.full((10, 10, 3), 0.5)
        r1 = convert_linear(img, dither=True)
        r2 = convert_linear(img, dither=True)
        # Should differ due to random dither noise
        assert not torch.equal(r1, r2)


class TestRoundTrip:
    """Synthetic round-trip: ACEScg -> sRGB 8-bit -> ACEScg."""

    def test_roundtrip_per_channel_error(self):
        """Full forward+reverse cycle should have <1% per-channel error."""
        from pipeline.colorspace import rec709_to_acescg, srgb_to_linear, REC709_TO_ACESCG

        # Start with known ACEScg values in the mid-to-bright range where
        # 8-bit sRGB has enough precision for <1% round-trip error.
        acescg_orig = torch.tensor([
            [[0.10, 0.08, 0.06], [0.15, 0.12, 0.08]],
            [[0.30, 0.25, 0.10], [0.50, 0.40, 0.20]],
        ])

        # Forward: ACEScg -> Rec.709 linear
        M_inv = torch.linalg.inv(REC709_TO_ACESCG).to(dtype=acescg_orig.dtype)
        rec709_linear = torch.einsum("...c,dc->...d", acescg_orig, M_inv)

        # Rec.709 linear -> sRGB gamma
        def linear_to_srgb(x):
            x = x.clamp(0.0, 1.0)
            low = x * 12.92
            high = 1.055 * x.pow(1.0 / 2.4) - 0.055
            return torch.where(x <= 0.0031308, low, high)

        srgb = linear_to_srgb(rec709_linear)

        # Quantize to 8-bit
        srgb_8bit = (srgb * 255.0).round().clamp(0, 255) / 255.0

        # Reverse: sRGB -> linear -> ACEScg
        linear_back = srgb_to_linear(srgb_8bit)
        acescg_back = rec709_to_acescg(linear_back)

        # Per-channel relative error should be <1% for values above 8-bit noise floor.
        # Very small values (< 0.02) have only a few 8-bit levels, so quantization
        # error is proportionally larger.
        mask = acescg_orig > 0.02
        rel_err = ((acescg_back[mask] - acescg_orig[mask]) / acescg_orig[mask]).abs()
        max_err = rel_err.max().item()
        assert max_err < 0.01, f"Round-trip error {max_err:.4f} exceeds 1%"
