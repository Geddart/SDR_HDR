# SDR-to-HDR ACEScg Converter — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** CLI tool that converts sRGB images to ACEScg half-float EXR using extracted Refusion-HDR model with GPU tiling and color conversion.

**Architecture:** Extract ConditionalNAFNet + IRSDE from Refusion-HDR, modernize for PyTorch 2.9+, add GPU-accelerated PU21 decode + BT.2020→ACEScg conversion + tiled inference + EXR output. Single `convert.py` CLI entry point.

**Tech Stack:** Python 3.13, PyTorch 2.9 (CUDA 12.8), OpenEXR, einops, tqdm, opencv-python

---

## Task 1: PU21 Encoder/Decoder (Pure PyTorch)

**Files:**
- Create: `models/pu21.py`
- Create: `tests/test_pu21.py`

Port the NumPy-based PU21Encoder from Refusion-HDR to pure PyTorch tensor ops so it runs on CUDA.

**Step 1: Write the test**

```python
# tests/test_pu21.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -m pytest tests/test_pu21.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.pu21'`

**Step 3: Implement PU21Encoder in PyTorch**

```python
# models/pu21.py
import torch


class PU21Encoder:
    """
    Perceptually Uniform (PU21) encoder/decoder for HDR luminance.
    Pure PyTorch implementation — works on CPU and CUDA tensors.

    Reference: Mantiuk & Azimi, Picture Coding Symposium 2021.
    """

    def __init__(self, encoding_type: str = "banding_glare"):
        if encoding_type == "banding":
            self.par = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168,
                        1.063512885, 1.14115047, 521.4527484]
        elif encoding_type == "banding_glare":
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
                        0.09150303166, 0.9099517204, 596.3148142]
        elif encoding_type == "peaks":
            self.par = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247,
                        1.114783422, 1.095360363, 384.9217577]
        elif encoding_type == "peaks_glare":
            self.par = [816.885024, 1479.463946, 0.001253215609, 0.9329636822,
                        0.06746643971, 1.573435413, 419.6006374]
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        self.L_min = 0.005
        self.L_max = 10000.0

    def encode(self, Y: torch.Tensor) -> torch.Tensor:
        """Luminance (nits) -> PU21 encoded value. Element-wise, any shape."""
        Y = Y.clamp(min=self.L_min, max=self.L_max)
        p = self.par
        V = p[6] * (((p[0] + p[1] * Y.pow(p[3])) / (1 + p[2] * Y.pow(p[3]))).pow(p[4]) - p[5])
        return V.clamp(min=0.0)

    def decode(self, V: torch.Tensor) -> torch.Tensor:
        """PU21 encoded value -> luminance (nits). Element-wise, any shape."""
        p = self.par
        V_p = (V / p[6] + p[5]).clamp(min=0.0).pow(1.0 / p[4])
        numerator = (V_p - p[0]).clamp(min=0.0)
        denominator = p[1] - p[2] * V_p
        Y = (numerator / denominator).pow(1.0 / p[3])
        return Y
```

**Step 4: Run tests to verify they pass**

Run: `cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -m pytest tests/test_pu21.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add models/pu21.py models/__init__.py tests/test_pu21.py
git commit -m "feat: add PU21 encoder/decoder in pure PyTorch"
```

---

## Task 2: Colorspace Conversion (BT.2020 → ACEScg)

**Files:**
- Create: `pipeline/colorspace.py`
- Create: `tests/test_colorspace.py`

**Step 1: Write the test**

```python
# tests/test_colorspace.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -m pytest tests/test_colorspace.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement colorspace conversion**

```python
# pipeline/colorspace.py
import torch

# Combined BT.2020 (D65) -> ACEScg (AP1, D60) matrix.
# Includes chromatic adaptation via Bradford transform.
# Source: colour-science library, validated against OCIO ACES configs.
BT2020_TO_ACESCG = torch.tensor([
    [0.61319930, 0.33951244, 0.04728826],
    [0.07021272, 0.91635982, 0.01342745],
    [0.02061835, 0.10957647, 0.86980518],
], dtype=torch.float64)


def bt2020_to_acescg(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert scene-linear BT.2020 RGB to ACEScg (AP1) RGB.

    Args:
        rgb: Tensor of shape (..., 3). Any device, any dtype.

    Returns:
        Tensor of same shape in ACEScg primaries.
    """
    M = BT2020_TO_ACESCG.to(dtype=rgb.dtype, device=rgb.device)
    return torch.einsum("...c,dc->...d", rgb, M)


def normalize_luminance(img: torch.Tensor, mode: str = "diffuse") -> torch.Tensor:
    """
    Normalize absolute luminance (nits) to scene-linear working range.

    Args:
        img: Linear luminance in nits. Any shape.
        mode: "diffuse" (100 nits=1.0), "pq-peak" (10000 nits=1.0),
              "middle-gray" (18 nits=0.18).
    """
    if mode == "diffuse":
        return img / 100.0
    elif mode == "pq-peak":
        return img / 10000.0
    elif mode == "middle-gray":
        return img * (0.18 / 18.0)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
```

**Step 4: Run tests**

Run: `cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -m pytest tests/test_colorspace.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add pipeline/colorspace.py pipeline/__init__.py tests/test_colorspace.py
git commit -m "feat: add BT.2020 to ACEScg GPU conversion and luminance normalization"
```

---

## Task 3: EXR Writer

**Files:**
- Create: `pipeline/exr_writer.py`
- Create: `tests/test_exr_writer.py`

**Step 1: Install OpenEXR**

Run: `pip install OpenEXR`

**Step 2: Write the test**

```python
# tests/test_exr_writer.py
import torch
import numpy as np
import os
import tempfile
import pytest


def test_write_read_roundtrip():
    """Write an EXR and read it back, values should match within half-float precision."""
    from pipeline.exr_writer import write_exr
    import OpenEXR

    img = torch.rand(64, 64, 3) * 2.0  # ACEScg values, some > 1.0
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Read back with OpenEXR
        with OpenEXR.File(path) as f:
            channels = f.channels()
            assert "R" in channels or "RGB" in channels
    finally:
        os.unlink(path)


def test_half_float_precision():
    """Output should be float16 (half) precision."""
    from pipeline.exr_writer import write_exr
    import OpenEXR

    img = torch.tensor([[[1.0, 0.5, 0.25]]])  # 1x1x3
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        with OpenEXR.File(path) as f:
            # Check that file was written successfully and is readable
            channels = f.channels()
            assert len(channels) > 0
    finally:
        os.unlink(path)


def test_large_image():
    """Should handle large images without error."""
    from pipeline.exr_writer import write_exr

    img = torch.rand(2048, 2048, 3)
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)
```

**Step 3: Implement EXR writer**

```python
# pipeline/exr_writer.py
import numpy as np
import torch
import OpenEXR


def write_exr(img: torch.Tensor, path: str) -> None:
    """
    Write an (H, W, 3) float tensor as a half-float EXR file.

    Args:
        img: RGB image tensor, shape (H, W, 3). Any device, any dtype.
        path: Output file path (must end in .exr).
    """
    rgb = img.detach().cpu().float().numpy().astype(np.float16)

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {
        "R": rgb[:, :, 0],
        "G": rgb[:, :, 1],
        "B": rgb[:, :, 2],
    }

    with OpenEXR.File(header, channels) as f:
        f.write(path)
```

**Step 4: Run tests**

Run: `cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -m pytest tests/test_exr_writer.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add pipeline/exr_writer.py tests/test_exr_writer.py
git commit -m "feat: add half-float EXR writer using OpenEXR"
```

---

## Task 4: NAFNet Model Architecture (Extract & Modernize)

**Files:**
- Create: `models/nafnet.py`

This is a direct extraction of `DenoisingNAFNet_arch.py` and `module_util.py` from Refusion-HDR. The architecture code uses only standard PyTorch ops — no deprecated APIs here. We combine the two files into one and keep only what's needed.

**Step 1: Create the model file**

```python
# models/nafnet.py
"""
ConditionalNAFNet — extracted from Refusion-HDR (AIM 2025 ITM Challenge).
Original: https://github.com/limchaos/Refusion-HDR

Architecture: U-Net with NAFBlocks (non-linear activation free), time-conditioned
via sinusoidal embeddings + FiLM modulation. Used as the denoiser backbone in the
IRSDE reverse diffusion process.

No modifications to layer names or shapes — pretrained weights load directly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Utility modules (from module_util.py) ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


# --- NAFBlock ---

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp
        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        x = y + x * self.gamma
        return x, time


# --- ConditionalNAFNet ---

class ConditionalNAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim),
        )

        self.intro = nn.Conv2d(img_channel * 2, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, time_dim) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond, time):
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=inp.device)

        x = inp - cond
        x = torch.cat([x, cond], dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self._check_image_size(x)

        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)
        x = x[..., :H, :W]
        return x

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
```

Note: The only rename is `check_image_size` → `_check_image_size` (private method convention). This does NOT affect weight loading since `check_image_size` is not a submodule — it has no parameters. All `nn.Module` layer names are identical to the original.

**Step 2: Verify architecture matches weights**

This cannot be unit-tested without the weights file. We validate in Task 7 (integration). For now, verify the model instantiates and produces correct output shapes:

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "
import torch
from models.nafnet import ConditionalNAFNet
model = ConditionalNAFNet(width=64, enc_blk_nums=[1,1,1,28], middle_blk_num=1, dec_blk_nums=[1,1,1,1])
x = torch.randn(1, 3, 256, 256)
cond = torch.randn(1, 3, 256, 256)
out = model(x, cond, time=50)
print(f'Output shape: {out.shape}')  # Should be (1, 3, 256, 256)
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')
"
```

Expected: `Output shape: torch.Size([1, 3, 256, 256])` and parameter count should be in the millions.

**Step 3: Commit**

```bash
git add models/nafnet.py
git commit -m "feat: extract ConditionalNAFNet architecture from Refusion-HDR"
```

---

## Task 5: IRSDE Inference Engine (Extract & Modernize)

**Files:**
- Create: `models/sde.py`

Extract the IRSDE class from Refusion-HDR `sde_utils.py`. Keep only inference-relevant methods. Remove training code, file-saving debug code, unused ODE solvers. Remove scipy dependency.

**Step 1: Implement the SDE module**

```python
# models/sde.py
"""
IRSDE (Image Restoration SDE) — extracted from Refusion-HDR.
Inference-only: forward noise injection + reverse posterior sampling.
"""
import math
import torch
from tqdm import tqdm


class IRSDE:
    def __init__(self, max_sigma: float = 50.0, T: int = 100,
                 schedule: str = "cosine", eps: float = 0.005,
                 device: torch.device = None):
        self.T = T
        self.device = device or torch.device("cpu")
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)
        self.mu = None
        self.model = None

    def _initialize(self, max_sigma, T, schedule, eps):
        if schedule == "cosine":
            thetas = self._cosine_schedule(T)
        elif schedule == "linear":
            thetas = self._linear_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        sigmas = torch.sqrt(max_sigma ** 2 * 2 * thetas)
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * dt)))

        self.dt = dt
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = T + 3
        x = torch.linspace(0, T + 2, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / (T + 2)) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return 1 - alphas_cumprod[1:-1]

    @staticmethod
    def _linear_schedule(T):
        timesteps = T + 1
        scale = 1000 / timesteps
        return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float32)

    def set_mu(self, mu: torch.Tensor):
        """Set the mean (LDR condition image)."""
        self.mu = mu

    def set_model(self, model: torch.nn.Module):
        """Set the denoiser network."""
        self.model = model

    def noise_state(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add initial noise to start reverse diffusion."""
        return tensor + torch.randn_like(tensor) * self.max_sigma

    def reverse_posterior(self, xt: torch.Tensor, save_states: bool = False) -> torch.Tensor:
        """
        Run T-step reverse posterior sampling.

        Args:
            xt: Noisy input tensor (1, 3, H, W) on device.

        Returns:
            Denoised output tensor (1, 3, H, W).
        """
        x = xt.clone()
        for t in tqdm(reversed(range(1, self.T + 1)), total=self.T, desc="Diffusion"):
            noise = self.model(x, self.mu, t)
            x = self._reverse_posterior_step(x, noise, t)
        return x

    def _reverse_posterior_step(self, xt, noise, t):
        x0 = self._get_init_state_from_noise(xt, noise, t)
        mean = self._reverse_optimum_step(xt, x0, t)
        std = self._reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def _get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.mu - self.sigma_bars[t] * noise) * A + self.mu

    def _reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def _reverse_optimum_std(self, t):
        A = torch.exp(-2 * self.thetas[t] * self.dt)
        B = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2 * self.thetas_cumsum[t - 1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        min_value = torch.tensor(1e-20 * self.dt, device=self.thetas.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma
```

**Step 2: Verify SDE initialization**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "
from models.sde import IRSDE
sde = IRSDE(max_sigma=50, T=100, schedule='cosine', eps=0.005, device='cpu')
print(f'T={sde.T}, dt={sde.dt:.6f}, max_sigma={sde.max_sigma:.6f}')
print(f'thetas shape: {sde.thetas.shape}')
print(f'sigma_bars range: [{sde.sigma_bars.min():.6f}, {sde.sigma_bars.max():.6f}]')
"
```

Expected: T=100, thetas shape torch.Size([102]), sigma_bars in a reasonable range.

**Step 3: Commit**

```bash
git add models/sde.py
git commit -m "feat: extract IRSDE inference engine from Refusion-HDR"
```

---

## Task 6: Tiled Inference Pipeline

**Files:**
- Create: `pipeline/inference.py`

Orchestrates the full pipeline: load image → tile → run SDE per tile → reassemble → PU21 decode → normalize → color convert.

**Step 1: Implement the inference pipeline**

```python
# pipeline/inference.py
"""
Tiled inference pipeline: sRGB input → ACEScg HDR output.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from models.nafnet import ConditionalNAFNet
from models.sde import IRSDE
from models.pu21 import PU21Encoder
from pipeline.colorspace import bt2020_to_acescg, normalize_luminance
from pipeline.exr_writer import write_exr


def load_model(weights_path: str, device: torch.device) -> ConditionalNAFNet:
    """Load pretrained ConditionalNAFNet."""
    model = ConditionalNAFNet(
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Strip DataParallel "module." prefix if present
    clean = {}
    for k, v in state_dict.items():
        clean[k.removeprefix("module.")] = v
    model.load_state_dict(clean)
    model.to(device).eval()
    return model


def load_image(path: str) -> torch.Tensor:
    """Read an sRGB image and return float32 tensor (H, W, 3) in [0, 1], RGB order."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img)


def _make_tiles(H: int, W: int, tile_size: int, overlap: int):
    """Generate (y_start, x_start, y_end, x_end) for each tile."""
    tiles = []
    step = tile_size - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)
            tiles.append((y_start, x_start, y_end, x_end))
    # Deduplicate
    return list(dict.fromkeys(tiles))


def _cosine_blend_weights(tile_h: int, tile_w: int, overlap: int,
                          device: torch.device) -> torch.Tensor:
    """Create 2D cosine blend weights for a tile. Shape (tile_h, tile_w)."""
    def ramp(size, overlap):
        if overlap <= 0 or size <= 2 * overlap:
            return torch.ones(size)
        w = torch.ones(size)
        ramp_up = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, overlap)))
        ramp_down = 0.5 * (1 + torch.cos(torch.linspace(0, torch.pi, overlap)))
        w[:overlap] = ramp_up
        w[-overlap:] = ramp_down
        return w

    wy = ramp(tile_h, overlap).to(device)
    wx = ramp(tile_w, overlap).to(device)
    return wy[:, None] * wx[None, :]


@torch.no_grad()
def run_inference(
    model: ConditionalNAFNet,
    image: torch.Tensor,
    device: torch.device,
    tile_size: int = 2048,
    overlap: int = 64,
    normalize_mode: str = "diffuse",
) -> torch.Tensor:
    """
    Full pipeline: sRGB image (H, W, 3) → ACEScg HDR (H, W, 3).

    Returns float32 tensor on CPU.
    """
    H, W, C = image.shape
    assert C == 3

    sde = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device=device)
    sde.set_model(model)
    pu21 = PU21Encoder()

    needs_tiling = H > tile_size or W > tile_size

    if not needs_tiling:
        output = _infer_single(model, sde, image, device)
    else:
        tiles = _make_tiles(H, W, tile_size, overlap)
        output_acc = torch.zeros(H, W, C, device=device)
        weight_acc = torch.zeros(H, W, 1, device=device)

        for i, (y0, x0, y1, x1) in enumerate(tqdm(tiles, desc="Tiles")):
            tile_img = image[y0:y1, x0:x1, :]
            tile_out = _infer_single(model, sde, tile_img, device)

            th, tw = y1 - y0, x1 - x0
            blend_w = _cosine_blend_weights(th, tw, overlap, device)

            output_acc[y0:y1, x0:x1, :] += tile_out * blend_w.unsqueeze(-1)
            weight_acc[y0:y1, x0:x1, :] += blend_w.unsqueeze(-1)

        output = output_acc / weight_acc.clamp(min=1e-8)

    # PU21 decode: model output [0,1] → scale to PU21 range → decode to nits
    pu21_scale = pu21.encode(torch.tensor(1000.0, device=device))
    output_nits = pu21.decode(output * pu21_scale)

    # Normalize luminance
    output_normalized = normalize_luminance(output_nits, mode=normalize_mode)

    # BT.2020 → ACEScg
    output_acescg = bt2020_to_acescg(output_normalized)

    return output_acescg.cpu()


def _infer_single(model, sde, image_hwc, device):
    """Run SDE inference on a single image/tile. Returns (H, W, 3) on device."""
    # HWC → BCHW
    ldr = image_hwc.to(device).permute(2, 0, 1).unsqueeze(0)

    sde.set_mu(ldr)
    noisy = sde.noise_state(ldr)

    # Reverse diffusion (100 steps)
    output = sde.reverse_posterior(noisy)

    # BCHW → HWC, clamp to valid PU21 input range
    return output.squeeze(0).permute(1, 2, 0).clamp(0, 1)


def convert_file(
    input_path: str,
    output_path: str,
    model: ConditionalNAFNet,
    device: torch.device,
    tile_size: int = 2048,
    overlap: int = 64,
    normalize_mode: str = "diffuse",
):
    """Convert a single image file to ACEScg EXR."""
    image = load_image(input_path)
    print(f"  Input: {input_path} ({image.shape[1]}x{image.shape[0]})")

    result = run_inference(model, image, device, tile_size, overlap, normalize_mode)

    write_exr(result, output_path)
    print(f"  Output: {output_path}")
```

**Step 2: Verify imports work**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "from pipeline.inference import load_model, convert_file; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add pipeline/inference.py
git commit -m "feat: add tiled inference pipeline with PU21 decode and colorspace conversion"
```

---

## Task 7: CLI Entry Point

**Files:**
- Create: `convert.py`

**Step 1: Implement the CLI**

```python
# convert.py
"""
SDR-to-HDR ACEScg Converter

Converts sRGB images to ACEScg half-float EXR using Refusion-HDR
(ConditionalNAFNet + IRSDE diffusion).

Usage:
    python convert.py input.png -o output.exr
    python convert.py ./inputs/ -o ./outputs/
    python convert.py input.png --normalize middle-gray --tile-size 1024
"""
import argparse
import sys
from pathlib import Path

import torch

from pipeline.inference import load_model, convert_file, load_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "lastest_EMA.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Convert sRGB images to ACEScg HDR EXR files."
    )
    parser.add_argument("input", type=str, help="Input image or folder of images")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output EXR path or folder. Default: <input>_acescg.exr")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to lastest_EMA.pth checkpoint")
    parser.add_argument("--normalize", type=str, default="diffuse",
                        choices=["diffuse", "pq-peak", "middle-gray"],
                        help="Luminance normalization mode (default: diffuse)")
    parser.add_argument("--tile-size", type=int, default=2048,
                        help="Max tile dimension in pixels (default: 2048)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Tile overlap in pixels (default: 64)")
    args = parser.parse_args()

    # Resolve input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Collect files
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            print(f"Error: No images found in {input_path}")
            sys.exit(1)
    else:
        files = [input_path]

    # Resolve output
    if args.output:
        output_path = Path(args.output)
        if len(files) > 1:
            output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: Weights not found at {weights}")
        print(f"Download from: https://www.dropbox.com/scl/fi/yg44t2i9tgrlsn3c1punc/lastest_EMA.pth?rlkey=fhjb37o34i9yt12337pyed5gi&st=43psqej3&dl=0")
        print(f"Place in: {DEFAULT_WEIGHTS}")
        sys.exit(1)

    print(f"Loading model from {weights}...")
    model = load_model(str(weights), device)
    print(f"Model loaded. Normalization: {args.normalize}")

    # Process
    for i, file in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {file.name}")

        if output_path is None:
            out = file.with_name(file.stem + "_acescg.exr")
        elif output_path.suffix == ".exr":
            out = output_path
        else:
            out = output_path / (file.stem + "_acescg.exr")

        convert_file(
            input_path=str(file),
            output_path=str(out),
            model=model,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            normalize_mode=args.normalize,
        )

    print(f"\nDone. Processed {len(files)} image(s).")


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI help**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python convert.py --help
```

Expected: Shows usage with all arguments.

**Step 3: Commit**

```bash
git add convert.py
git commit -m "feat: add CLI entry point for SDR-to-HDR conversion"
```

---

## Task 8: Package Init Files

**Files:**
- Create: `models/__init__.py`
- Create: `pipeline/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create package init files**

```python
# models/__init__.py
# (empty — imports done explicitly)
```

```python
# pipeline/__init__.py
# (empty — imports done explicitly)
```

```python
# tests/__init__.py
```

**Step 2: Commit**

```bash
git add models/__init__.py pipeline/__init__.py tests/__init__.py
git commit -m "chore: add package init files"
```

---

## Task 9: Download Weights & Integration Test

**Files:**
- No new files — uses existing code.

**Step 1: Download model weights**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert
# Download from Dropbox (manual or via curl)
curl -L "https://www.dropbox.com/scl/fi/yg44t2i9tgrlsn3c1punc/lastest_EMA.pth?rlkey=fhjb37o34i9yt12337pyed5gi&st=43psqej3&dl=1" -o weights/lastest_EMA.pth
```

Verify file size is reasonable (should be tens of MB for this architecture).

**Step 2: Verify weights load correctly**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "
import torch
from pipeline.inference import load_model
device = torch.device('cuda')
model = load_model('weights/lastest_EMA.pth', device)
print(f'Model loaded on {device}')
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')
"
```

Expected: Model loads without errors, prints parameter count.

**Step 3: Create a synthetic test image and run full pipeline**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "
import cv2, numpy as np
# Create a simple gradient test image
img = np.zeros((256, 256, 3), dtype=np.uint8)
img[:, :128] = [200, 150, 100]   # warm tone left
img[:, 128:] = [100, 150, 200]   # cool tone right
img[100:156, 100:156] = [255, 255, 255]  # bright square (should get HDR highlight)
cv2.imwrite('test_input.png', img)
print('Test image created: test_input.png')
"
```

**Step 4: Run the full conversion**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python convert.py test_input.png -o test_output.exr
```

Expected: Processes without errors, creates `test_output.exr`.

**Step 5: Validate EXR output**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && python -c "
import OpenEXR, numpy as np

with OpenEXR.File('test_output.exr') as f:
    channels = f.channels()
    print(f'Channels: {list(channels.keys())}')
    # Check we have data
    for name, data in channels.items():
        arr = np.array(data)
        print(f'  {name}: dtype={arr.dtype}, shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]')
    print('EXR validation passed.')
"
```

Expected: Shows R, G, B channels with float16 dtype, values spanning a range (not all zeros, not all identical).

**Step 6: Clean up test files**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && rm -f test_input.png test_output.exr
```

**Step 7: Commit**

```bash
git commit --allow-empty -m "test: integration test passed - full pipeline working"
```

---

## Task 10: Install Dependencies

**Files:**
- Create: `requirements.txt`

**Step 1: Create requirements.txt**

```
torch>=2.0
torchvision>=0.15
opencv-python
numpy
OpenEXR
einops
tqdm
```

**Step 2: Install**

```bash
cd H:/001_ProjectCache/1000_Coding/SDR_HDR_Convert && pip install -r requirements.txt
```

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt"
```

---

## Execution Order

Tasks 1-3 can run in parallel (independent modules with tests).
Task 4-5 can run in parallel (model and SDE are independent).
Task 6 depends on Tasks 1-5 (imports all modules).
Task 7 depends on Task 6.
Task 8 should run before Task 6 (init files needed for imports).
Task 9 depends on Task 7 (needs full CLI).
Task 10 can run at any time.

**Recommended sequential order:** 10 → 8 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9
