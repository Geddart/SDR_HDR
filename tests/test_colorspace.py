import torch
import pytest


def test_white_point_preserved():
    """D65 white in BT.2020 should map to near-white in ACEScg."""
    from pipeline.colorspace import bt2020_to_acescg
    # Equal-energy white (1,1,1) in BT.2020 should sum to ~(1,1,1) in ACEScg
    white = torch.tensor([1.0, 1.0, 1.0])
    result = bt2020_to_acescg(white)
    # Row sums of the matrix should be ~1.0 (white preservation)
    assert result.sum().item() == pytest.approx(3.0, abs=0.01)


def test_known_red_primary():
    """Pure BT.2020 red should map to predominantly R in ACEScg."""
    from pipeline.colorspace import bt2020_to_acescg
    red_bt2020 = torch.tensor([1.0, 0.0, 0.0])
    result = bt2020_to_acescg(red_bt2020)
    # R channel should be dominant
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_image_batch():
    """Should handle (H, W, 3) image tensors."""
    from pipeline.colorspace import bt2020_to_acescg
    img = torch.rand(64, 64, 3)
    result = bt2020_to_acescg(img)
    assert result.shape == (64, 64, 3)


def test_gpu():
    """Should work on CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    from pipeline.colorspace import bt2020_to_acescg
    img = torch.rand(64, 64, 3, device="cuda")
    result = bt2020_to_acescg(img)
    assert result.device.type == "cuda"
    assert result.shape == (64, 64, 3)


def test_normalize_diffuse():
    """100 nits input should produce 1.0 output with diffuse normalization."""
    from pipeline.colorspace import normalize_luminance
    img = torch.full((2, 2, 3), 100.0)  # 100 nits everywhere
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
