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
    # BGR -> RGB
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
    Full pipeline: sRGB image (H, W, 3) -> ACEScg HDR (H, W, 3).

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

    # PU21 decode: model output [0,1] -> scale to PU21 range -> decode to nits
    pu21_scale = pu21.encode(torch.tensor(1000.0, device=device))
    output_nits = pu21.decode(output * pu21_scale)

    # Normalize luminance
    output_normalized = normalize_luminance(output_nits, mode=normalize_mode)

    # BT.2020 -> ACEScg
    output_acescg = bt2020_to_acescg(output_normalized)

    return output_acescg.cpu()


def _infer_single(model, sde, image_hwc, device):
    """Run SDE inference on a single image/tile. Returns (H, W, 3) on device."""
    # HWC -> BCHW
    ldr = image_hwc.to(device).permute(2, 0, 1).unsqueeze(0)

    sde.set_mu(ldr)
    noisy = sde.noise_state(ldr)

    # Reverse diffusion (100 steps)
    output = sde.reverse_posterior(noisy)

    # BCHW -> HWC, clamp to valid PU21 input range
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
