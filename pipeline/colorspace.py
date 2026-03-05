import torch

# Rec.709 / sRGB (D65) -> ACEScg (AP1, D60) matrix.
# Includes chromatic adaptation via Bradford transform.
# Computed from CIE xy primaries + Bradford CAT D65->D60.
REC709_TO_ACESCG = torch.tensor([
    [0.6130974024, 0.3395231462, 0.0473794514],
    [0.0701937225, 0.9163538791, 0.0134523985],
    [0.0206155929, 0.1095697729, 0.8698146342],
], dtype=torch.float64)

# BT.2020 (D65) -> ACEScg (AP1, D60) matrix.
# Near-identity since BT.2020 and ACEScg (AP1) have similar primaries.
BT2020_TO_ACESCG = torch.tensor([
    [0.9748949779, 0.0195991086, 0.0055059134],
    [0.0021795628, 0.9955354689, 0.0022849683],
    [0.0047972397, 0.0245320166, 0.9706707437],
], dtype=torch.float64)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """
    Decode sRGB EOTF to linear Rec.709.

    Uses the piecewise sRGB transfer function (IEC 61966-2-1).
    Input and output in [0, 1] range.
    """
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055).clamp(min=0.0).pow(2.4)
    return torch.where(rgb <= 0.04045, low, high)


def rec709_to_acescg(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert linear Rec.709 / sRGB RGB to ACEScg (AP1) RGB.

    Args:
        rgb: Tensor of shape (..., 3). Any device, any dtype.

    Returns:
        Tensor of same shape in ACEScg primaries.
    """
    M = REC709_TO_ACESCG.to(dtype=rgb.dtype, device=rgb.device)
    return torch.einsum("...c,dc->...d", rgb, M)


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


def inverse_tonemap(
    linear: torch.Tensor,
    peak: float = 150.0,
    gain: float = 3.0,
    power: float = 2.0,
) -> torch.Tensor:
    """
    SDR-to-HDR expansion with mid-tone lift and highlight rolloff.

    Maps linear [0, 1] to HDR [0, peak]:
        hdr = linear * (gain + (peak - gain) * linear^power)

    - Low values behave as linear * gain (lifts mid-tones)
    - At linear=1.0, output equals peak exactly
    - Higher power keeps shadows/midtones closer to linear*gain,
      concentrating the HDR expansion near white.

    Args:
        linear: Scene-linear values in [0, 1]. Any shape.
        peak: Maximum output value. sRGB white (1.0) maps to this.
        gain: Mid-tone multiplier. Controls brightness of non-highlight pixels.
        power: Exponent controlling highlight rolloff steepness. Default 2.0.
    """
    L = linear.clamp(0.0, 1.0)
    return L * (gain + (peak - gain) * L.pow(power))


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
