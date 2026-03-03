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
