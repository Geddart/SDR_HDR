import torch
import pytest


def test_rec709_white_point_preserved():
    """D65 white in Rec.709 should map to near-white in ACEScg."""
    from pipeline.colorspace import rec709_to_acescg
    white = torch.tensor([1.0, 1.0, 1.0])
    result = rec709_to_acescg(white)
    assert result.sum().item() == pytest.approx(3.0, abs=0.01)


def test_bt2020_white_point_preserved():
    """D65 white in BT.2020 should map to near-white in ACEScg."""
    from pipeline.colorspace import bt2020_to_acescg
    white = torch.tensor([1.0, 1.0, 1.0])
    result = bt2020_to_acescg(white)
    assert result.sum().item() == pytest.approx(3.0, abs=0.01)


def test_rec709_red_primary():
    """Pure Rec.709 red should map to predominantly R in ACEScg."""
    from pipeline.colorspace import rec709_to_acescg
    red = torch.tensor([1.0, 0.0, 0.0])
    result = rec709_to_acescg(red)
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_image_batch():
    """Should handle (H, W, 3) image tensors."""
    from pipeline.colorspace import rec709_to_acescg
    img = torch.rand(64, 64, 3)
    result = rec709_to_acescg(img)
    assert result.shape == (64, 64, 3)


def test_gpu():
    """Should work on CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    from pipeline.colorspace import rec709_to_acescg
    img = torch.rand(64, 64, 3, device="cuda")
    result = rec709_to_acescg(img)
    assert result.device.type == "cuda"
    assert result.shape == (64, 64, 3)


def test_srgb_to_linear():
    """sRGB decode should match known values."""
    from pipeline.colorspace import srgb_to_linear
    # sRGB 0.5 -> linear ~0.214
    result = srgb_to_linear(torch.tensor([0.0, 0.5, 1.0]))
    assert result[0].item() == pytest.approx(0.0, abs=1e-6)
    assert result[1].item() == pytest.approx(0.214, abs=0.001)
    assert result[2].item() == pytest.approx(1.0, abs=1e-6)


def test_normalize_diffuse():
    """100 nits input should produce 1.0 output with diffuse normalization."""
    from pipeline.colorspace import normalize_luminance
    img = torch.full((2, 2, 3), 100.0)
    result = normalize_luminance(img, mode="diffuse")
    torch.testing.assert_close(result, torch.ones(2, 2, 3), rtol=1e-5, atol=1e-5)


def test_normalize_pq_peak():
    """10000 nits should produce 1.0 with pq-peak normalization."""
    from pipeline.colorspace import normalize_luminance
    img = torch.full((2, 2, 3), 10000.0)
    result = normalize_luminance(img, mode="pq-peak")
    torch.testing.assert_close(result, torch.ones(2, 2, 3), rtol=1e-5, atol=1e-5)


def test_normalize_middle_gray():
    """18 nits should produce 0.18 with middle-gray normalization."""
    from pipeline.colorspace import normalize_luminance
    img = torch.full((2, 2, 3), 18.0)
    result = normalize_luminance(img, mode="middle-gray")
    expected = torch.full((2, 2, 3), 0.18)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_inverse_tonemap_zero():
    """Black input should produce zero output."""
    from pipeline.colorspace import inverse_tonemap
    result = inverse_tonemap(torch.tensor([0.0]))
    assert result.item() == pytest.approx(0.0, abs=1e-8)


def test_inverse_tonemap_one_equals_peak():
    """Input 1.0 should map to exactly peak."""
    from pipeline.colorspace import inverse_tonemap
    for peak in [100.0, 150.0, 200.0]:
        result = inverse_tonemap(torch.tensor([1.0]), peak=peak, gain=3.0)
        assert result.item() == pytest.approx(peak, abs=0.01), f"peak={peak}"


def test_inverse_tonemap_midtone_gain():
    """Small values should behave approximately as linear * gain."""
    from pipeline.colorspace import inverse_tonemap
    x = torch.tensor([0.01])
    result = inverse_tonemap(x, peak=150.0, gain=3.0)
    # At x=0.01: gain + (peak-gain)*x^2 ≈ gain (since x^2 is tiny)
    expected = 0.01 * 3.0
    assert result.item() == pytest.approx(expected, rel=0.05)


def test_inverse_tonemap_monotonic():
    """Output should be monotonically increasing."""
    from pipeline.colorspace import inverse_tonemap
    x = torch.linspace(0, 1, 100)
    y = inverse_tonemap(x, peak=150.0, gain=3.0)
    diffs = y[1:] - y[:-1]
    assert (diffs >= 0).all()
