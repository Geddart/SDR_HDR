"""
Tiled inference pipeline: sRGB input -> ACEScg HDR output.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from models.nafnet import ConditionalNAFNet
from models.sde import IRSDE
from models.pu21 import PU21Encoder
from pipeline.colorspace import (
    srgb_to_linear, rec709_to_acescg, inverse_tonemap, normalize_luminance,
)
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
    clean = {}
    for k, v in state_dict.items():
        clean[k.removeprefix("module.")] = v
    model.load_state_dict(clean)
    model.to(device).eval()
    return model


def load_image(path: str):
    """
    Read an image and return (rgb, alpha) as float32 tensors.

    rgb: (H, W, 3) in [0, 1], RGB order (still sRGB gamma-encoded).
    alpha: (H, W) in [0, 1], or None if no alpha channel.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = None
    elif img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = None

    rgb = img.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb)
    alpha_tensor = torch.from_numpy(alpha) if alpha is not None else None
    return rgb_tensor, alpha_tensor


# ---------------------------------------------------------------------------
# Linear mode: sRGB -> linear Rec.709 -> ACEScg (no AI model)
# ---------------------------------------------------------------------------

def convert_linear(
    image: torch.Tensor,
    exposure: float = 0.0,
    peak: float = 150.0,
    gain: float = 3.0,
) -> torch.Tensor:
    """
    Convert sRGB image to ACEScg scene-linear with HDR highlight expansion.

    Pipeline: sRGB EOTF -> dither -> luminance-based inverse tone map -> Rec.709 -> ACEScg.

    The HDR curve is applied to luminance only, then RGB is scaled
    proportionally. This preserves chromaticity and avoids per-channel
    artifacts (solarization) that would appear when boosting exposure.

    Args:
        image: sRGB gamma-encoded (H, W, 3) in [0, 1].
        exposure: Exposure adjustment in stops (EV). 0 = no change.
        peak: Peak HDR value. sRGB white (1.0) maps to this in scene-linear.
        gain: Mid-tone multiplier for non-highlight brightness.

    Returns:
        ACEScg scene-linear (H, W, 3) float32.
    """
    linear = srgb_to_linear(image)

    # Triangular dither to break up 8-bit banding before nonlinear expansion.
    # TPDF noise at +/- 1 LSB of 8-bit source (1/255).
    noise = (torch.rand_like(linear) + torch.rand_like(linear) - 1.0) / 255.0
    linear = (linear + noise).clamp(0.0)

    # Rec.709 luminance
    Y = (0.2126 * linear[..., 0:1]
         + 0.7152 * linear[..., 1:2]
         + 0.0722 * linear[..., 2:3]).clamp(min=1e-10)

    # Apply HDR curve to luminance only, then scale RGB proportionally.
    # This preserves the original chromaticity and avoids solarization.
    Y_hdr = inverse_tonemap(Y, peak=peak, gain=gain)
    hdr = linear * (Y_hdr / Y)

    acescg = rec709_to_acescg(hdr)

    if exposure != 0.0:
        ev_gain = 2.0 ** exposure
        acescg = acescg * ev_gain

    return acescg


# ---------------------------------------------------------------------------
# Model mode: sRGB -> Refusion-HDR diffusion -> PU21 decode -> ACEScg
# ---------------------------------------------------------------------------

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
    tile_size: int = 1024,
    overlap: int = 64,
    normalize_mode: str = "diffuse",
    exposure: float = 0.0,
) -> torch.Tensor:
    """
    Full AI pipeline: sRGB image (H, W, 3) -> ACEScg HDR (H, W, 3).
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
        output_acc = torch.zeros(H, W, C)
        weight_acc = torch.zeros(H, W, 1)

        for i, (y0, x0, y1, x1) in enumerate(tqdm(tiles, desc="Tiles")):
            tile_img = image[y0:y1, x0:x1, :]
            tile_out = _infer_single(model, sde, tile_img, device)
            tile_out = tile_out.cpu()

            th, tw = y1 - y0, x1 - x0
            blend_w = _cosine_blend_weights(th, tw, overlap, "cpu")

            output_acc[y0:y1, x0:x1, :] += tile_out * blend_w.unsqueeze(-1)
            weight_acc[y0:y1, x0:x1, :] += blend_w.unsqueeze(-1)

            del tile_out
            torch.cuda.empty_cache()

        output = output_acc / weight_acc.clamp(min=1e-8)

    if output.device.type != "cpu":
        output = output.cpu()

    # PU21 decode: model output [0,1] -> scale to PU21 range -> decode to nits
    pu21_scale = pu21.encode(torch.tensor(1000.0))
    output_nits = pu21.decode(output * pu21_scale)

    # Normalize luminance
    output_normalized = normalize_luminance(output_nits, mode=normalize_mode)

    # Rec.709 -> ACEScg (model outputs in Rec.709/sRGB primaries)
    output_acescg = rec709_to_acescg(output_normalized)

    # Apply exposure
    if exposure != 0.0:
        gain = 2.0 ** exposure
        output_acescg = output_acescg * gain

    return output_acescg


def _infer_single(model, sde, image_hwc, device):
    """Run SDE inference on a single image/tile. Returns (H, W, 3) on device."""
    ldr = image_hwc.to(device).permute(2, 0, 1).unsqueeze(0)
    sde.set_mu(ldr)
    noisy = sde.noise_state(ldr)
    output = sde.reverse_posterior(noisy)
    return output.squeeze(0).permute(1, 2, 0).clamp(0, 1)


# ---------------------------------------------------------------------------
# File conversion entry point
# ---------------------------------------------------------------------------

def convert_file(
    input_path: str,
    output_path: str,
    device: torch.device,
    mode: str = "linear",
    model: ConditionalNAFNet = None,
    tile_size: int = 1024,
    overlap: int = 64,
    normalize_mode: str = "diffuse",
    exposure: float = 0.0,
    peak: float = 150.0,
    gain: float = 3.0,
):
    """Convert a single image file to ACEScg EXR. Preserves alpha if present."""
    image, alpha = load_image(input_path)
    print(f"  Input: {input_path} ({image.shape[1]}x{image.shape[0]}"
          f", alpha={'yes' if alpha is not None else 'no'})")

    if mode == "linear":
        result = convert_linear(image, exposure=exposure, peak=peak, gain=gain)
    elif mode == "model":
        if model is None:
            raise ValueError("Model mode requires a loaded model")
        result = run_inference(
            model, image, device, tile_size, overlap, normalize_mode, exposure,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Premultiply RGB by alpha for VFX-standard EXR output
    if alpha is not None:
        result = result * alpha.unsqueeze(-1)

    write_exr(result, output_path, alpha=alpha)
    print(f"  Output: {output_path}")
