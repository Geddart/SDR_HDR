# SDR to HDR — ACEScg Converter

Convert sRGB images to ACEScg half-float EXR with physically plausible HDR highlight expansion. Built for VFX artists who need HDR projection textures from SDR source material.

**What it does:** Takes an 8-bit sRGB image (PNG, JPEG, etc.) and produces an ACEScg scene-linear EXR where light sources hit 100-150+ and mid-tones sit in the 0.02-0.10 range — matching real camera plates from ARRI ALEXA or similar cinema cameras.

## Quick Start

```bash
# Install
git clone https://github.com/Geddart/SDR_HDR.git
cd SDR_HDR
pip install -r requirements.txt

# Convert a single image
python convert.py photo.png -o photo_acescg.exr

# Convert a folder
python convert.py ./textures/ -o ./textures_hdr/
```

Output files are half-float RGBA EXR with ZIP compression, premultiplied alpha, readable by Nuke, RV, 3ds Max, Houdini, and any ACES-aware application.

## How It Works

The default **linear mode** pipeline:

```
sRGB (8-bit) → EOTF decode → TPDF dither → luminance-based inverse tonemap → Rec.709 → ACEScg
```

1. **sRGB EOTF decode** — standard IEC 61966-2-1 piecewise transfer function
2. **Triangular dither** — breaks up 8-bit banding before the nonlinear expansion (prevents posterization when you push exposure in comp)
3. **Inverse tonemap on luminance** — expands dynamic range using `hdr = Y * (gain + (peak - gain) * Y²)`. Applied to Rec.709 luminance only, then RGB is scaled proportionally. This preserves chromaticity and prevents the solarization artifacts you'd get from per-channel expansion.
4. **Rec.709 → ACEScg matrix** — 3x3 color space conversion with Bradford chromatic adaptation (D65 → D60)
5. **Premultiply alpha** — standard VFX practice for EXR compositing

### Value Ranges (default settings)

| Content | ACEScg Value | Notes |
|---------|-------------|-------|
| Deep shadow | 0.001 - 0.005 | Near-black areas |
| Dark surface | 0.01 - 0.03 | Unlit stone, dark foliage |
| Lit surface | 0.03 - 0.11 | Sunlit wall, concrete |
| Bright surface | 0.5 - 5.0 | Reflective or bright materials |
| Light source | 100 - 150 | Street lights, lamps, sun reflections |

These ranges are calibrated against ARRI ALEXA 35 ACEScg plates where lit walls measure 0.02-0.11 and practical light sources measure 245-485.

## Parameters

```
python convert.py input -o output [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--peak` | `150` | Maximum HDR value. sRGB white (1.0) maps to this. |
| `--gain` | `3.0` | Mid-tone brightness multiplier. Higher = brighter shadows/mid-tones. |
| `--exposure` | `0.0` | Post-conversion exposure adjustment in stops. |
| `--mode` | `linear` | `linear` (fast, predictable) or `model` (AI diffusion, experimental). |

### Tuning Tips

**Too dark?** Increase `--gain` (try 4.0 or 5.0). This lifts mid-tones without changing the highlight peak.

**Highlights too hot?** Decrease `--peak` (try 100). This caps the maximum value light sources can reach.

**Need overall brighter/darker?** Use `--exposure` in stops. `--exposure 1` doubles everything, `--exposure -1` halves.

**Matching a specific plate?** Sample your plate's lit surfaces and light sources in RV/Nuke, then adjust `--gain` until mid-tones match and `--peak` until highlights match.

```bash
# Brighter mid-tones, moderate highlights
python convert.py input.png --gain 4.0 --peak 120

# Subtle HDR expansion (less aggressive)
python convert.py input.png --gain 2.0 --peak 80

# Aggressive HDR for environment lighting
python convert.py input.png --gain 3.0 --peak 200
```

## AI Model Mode (Experimental)

For cases where you need actual highlight reconstruction (not just expansion), there's an AI mode using the [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) diffusion model (ConditionalNAFNet + IRSDE).

```bash
# Download weights (292 MB)
# Place as: weights/lastest_EMA.pth

# Run with AI model
python convert.py input.png --mode model
```

This mode runs 100-step reverse diffusion to reconstruct clipped highlights, then decodes via PU21 to absolute luminance. It's significantly slower and requires a CUDA GPU with 8+ GB VRAM.

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `weights/lastest_EMA.pth` | Path to model checkpoint. |
| `--normalize` | `diffuse` | Luminance normalization: `diffuse` (100 nits=1.0), `pq-peak`, `middle-gray`. |
| `--tile-size` | `1024` | Max tile dimension. Lower = less VRAM usage. |
| `--overlap` | `64` | Tile overlap for seamless blending. |

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- OpenEXR, OpenCV, NumPy, einops, tqdm

```bash
pip install -r requirements.txt
```

### From Source

```bash
git clone https://github.com/Geddart/SDR_HDR.git
cd SDR_HDR
pip install -r requirements.txt
python -m pytest tests/  # verify installation
```

## Project Structure

```
SDR_HDR/
├── convert.py              # CLI entry point
├── pipeline/
│   ├── inference.py        # Conversion pipeline (linear + model modes)
│   ├── colorspace.py       # sRGB EOTF, Rec.709/BT.2020→ACEScg, inverse tonemap
│   └── exr_writer.py       # Half-float RGBA EXR output
├── models/
│   ├── nafnet.py           # ConditionalNAFNet architecture (Refusion-HDR)
│   ├── sde.py              # IRSDE inference engine
│   └── pu21.py             # PU21 perceptual encoder/decoder
├── tests/                  # pytest suite (16 tests)
├── weights/                # Model weights (gitignored)
└── requirements.txt
```

## Color Science Notes

- **Input**: sRGB (Rec.709 primaries, sRGB EOTF)
- **Output**: ACEScg (AP1 primaries, D60 white point, scene-linear)
- **Rec.709 → ACEScg matrix** includes Bradford chromatic adaptation D65 → D60
- **BT.2020 → ACEScg matrix** is near-identity (similar wide-gamut primaries)
- All color math is pure PyTorch — no OCIO or oiiotool dependency

## License

[MIT](LICENSE)

## Acknowledgments

- [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) — AI model architecture (AIM 2025 ITM Challenge)
- [PU21](https://doi.org/10.1109/TIP.2021.3070413) — Perceptually Uniform encoding (Mantiuk & Azimi 2021)
- Plate calibration reference from ARRI ALEXA 35 ACEScg production footage
