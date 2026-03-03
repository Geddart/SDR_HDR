import torch
import numpy as np
import pytest


def test_pu21_roundtrip():
    """encode then decode should return original luminance values."""
    from models.pu21 import PU21Encoder
    pu = PU21Encoder()
    # Test with known luminance values in nits
    Y = torch.tensor([0.005, 1.0, 18.0, 100.0, 1000.0, 10000.0])
    encoded = pu.encode(Y)
    decoded = pu.decode(encoded)
    torch.testing.assert_close(decoded, Y, rtol=1e-4, atol=1e-4)


def test_pu21_encode_1000():
    """PU21.encode(1000) should match the numpy reference implementation."""
    from models.pu21 import PU21Encoder
    pu = PU21Encoder()
    # Reference: numpy implementation with banding_glare params
    par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
           0.09150303166, 0.9099517204, 596.3148142]
    Y_np = np.array(1000.0)
    V_ref = par[6] * (((par[0] + par[1]*Y_np**par[3]) / (1 + par[2]*Y_np**par[3]))**par[4] - par[5])
    result = pu.encode(torch.tensor(1000.0))
    assert abs(result.item() - V_ref) < 0.01, f"Expected ~{V_ref}, got {result.item()}"


def test_pu21_gpu():
    """Should work on CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    from models.pu21 import PU21Encoder
    pu = PU21Encoder()
    Y = torch.tensor([100.0, 500.0, 1000.0], device="cuda")
    encoded = pu.encode(Y)
    assert encoded.device.type == "cuda"
    decoded = pu.decode(encoded)
    torch.testing.assert_close(decoded, Y, rtol=1e-4, atol=1e-4)


def test_pu21_batch_image():
    """Should handle (H, W, 3) shaped tensors."""
    from models.pu21 import PU21Encoder
    pu = PU21Encoder()
    Y = torch.rand(64, 64, 3) * 999 + 1  # random luminances 1-1000 nits
    encoded = pu.encode(Y)
    decoded = pu.decode(encoded)
    torch.testing.assert_close(decoded, Y, rtol=1e-3, atol=1e-3)
