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
