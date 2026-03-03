import numpy as np
import torch
import OpenEXR


def write_exr(img: torch.Tensor, path: str, alpha: torch.Tensor = None) -> None:
    """
    Write an (H, W, 3) float tensor as a half-float RGBA EXR file.

    Args:
        img: RGB image tensor, shape (H, W, 3). Any device, any dtype.
        path: Output file path (must end in .exr).
        alpha: Optional alpha tensor, shape (H, W). If None, alpha is set to 1.0.
    """
    rgb = img.detach().cpu().float().numpy().astype(np.float16)
    H, W = rgb.shape[:2]

    if alpha is not None:
        alpha_np = alpha.detach().cpu().float().numpy().astype(np.float16)
    else:
        alpha_np = np.ones((H, W), dtype=np.float16)

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {
        "R": np.ascontiguousarray(rgb[:, :, 0]),
        "G": np.ascontiguousarray(rgb[:, :, 1]),
        "B": np.ascontiguousarray(rgb[:, :, 2]),
        "A": np.ascontiguousarray(alpha_np),
    }

    with OpenEXR.File(header, channels) as f:
        f.write(path)
