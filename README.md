# SDR to HDR — ACEScg Converter

Turn any 8-bit image into a production-ready ACEScg EXR. AI-reconstructed highlights, physically plausible values, ready to light a scene.

Built for VFX artists who need real HDR projection textures from AI-generated images, photos, or any SDR source — without spending hours painting luminance by hand.

```bash
pip install .
sdr-hdr photo.png --mode model -o photo_acescg.exr
```

That's it. Weights download automatically. Tile size adapts to your GPU. Output is half-float RGBA EXR with premultiplied alpha — drops straight into Nuke, RV, 3ds Max, Houdini, or any ACES-aware app.

## What You Get

A street light in your sRGB source reads 1.0. In the output EXR, it reads 150+. A sunlit wall reads 0.05. Shadows sit at 0.002. Push exposure 6 stops in comp and it still holds up — no banding, no solarization, no clipped blobs where highlights used to be.

| Content | ACEScg Value |
|---------|-------------|
| Deep shadow | 0.001 - 0.005 |
| Dark surface (stone, foliage) | 0.01 - 0.03 |
| Lit surface (sunlit wall) | 0.03 - 0.11 |
| Bright surface | 0.5 - 5.0 |
| Light source | 100 - 150+ |

Calibrated against ARRI ALEXA 35 ACEScg plates. Lit walls: 0.02-0.11. Practicals: 245-485.

## Install

```bash
git clone https://github.com/Geddart/SDR_HDR.git
cd SDR_HDR
pip install .
```

Needs Python 3.10+ and PyTorch 2.0+. CUDA GPU with 8+ GB VRAM for model mode.

## Usage

```bash
# AI highlight reconstruction (recommended)
sdr-hdr photo.png --mode model -o photo_acescg.exr

# Batch convert a folder
sdr-hdr ./textures/ --mode model -o ./textures_hdr/

# Fast math-only conversion (no GPU needed)
sdr-hdr photo.png -o photo_acescg.exr
```

## Two Modes

### Model Mode (`--mode model`)

The real deal. A [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) diffusion model (ConditionalNAFNet + IRSDE) runs 100 reverse diffusion steps to reconstruct what was actually behind those clipped highlights. PU21 decodes the result to absolute luminance, then it's converted to ACEScg.

Weights (~292 MB) download automatically on first run. Tile size auto-adapts to your VRAM — works on 8 GB cards, flies on a 5090.

### Linear Mode (default)

Pure math, no AI. Fast, predictable, runs on CPU.

```
sRGB → EOTF decode → TPDF dither → luminance-based inverse tonemap → Rec.709 → ACEScg
```

Expands dynamic range using `hdr = Y * (gain + (peak - gain) * Y²)` on luminance only, then scales RGB proportionally. Preserves chromaticity, prevents solarization. Good when you need speed or don't have a GPU.

## Parameters

```
sdr-hdr input -o output [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `linear` | `linear` or `model` |
| `--exposure` | `0.0` | Exposure adjustment in stops |
| `--peak` | `150` | Max HDR value (linear mode) |
| `--gain` | `3.0` | Mid-tone multiplier (linear mode) |
| `--tile-size` | auto | Tile dimension, auto from VRAM (model mode) |
| `--overlap` | `64` | Tile overlap for blending (model mode) |
| `--normalize` | `diffuse` | `diffuse`, `pq-peak`, or `middle-gray` (model mode) |
| `--weights` | auto | Path to model checkpoint (model mode) |

### Tuning (Linear Mode)

```bash
# Brighter mid-tones, moderate highlights
sdr-hdr input.png --gain 4.0 --peak 120

# Subtle expansion
sdr-hdr input.png --gain 2.0 --peak 80

# Aggressive HDR for environment lighting
sdr-hdr input.png --gain 3.0 --peak 200
```

**Too dark?** Raise `--gain`. **Highlights too hot?** Lower `--peak`. **Overall shift?** Use `--exposure` in stops.

## How It Works

All color math is pure PyTorch on GPU — no OCIO, no oiiotool.

- **Input**: sRGB (Rec.709 primaries, sRGB EOTF)
- **Output**: ACEScg (AP1 primaries, D60 white point, scene-linear)
- **Color transform**: 3x3 matrix with Bradford chromatic adaptation D65 → D60
- **Dithering**: TPDF at ±1 LSB before nonlinear expansion — kills 8-bit banding
- **Alpha**: Premultiplied, preserved from source

## Project Structure

```
SDR_HDR/
├── convert.py              # CLI entry point
├── pyproject.toml          # pip install .
├── pipeline/
│   ├── inference.py        # Conversion pipeline (linear + model)
│   ├── colorspace.py       # sRGB EOTF, Rec.709→ACEScg, inverse tonemap
│   ├── exr_writer.py       # Half-float RGBA EXR output
│   └── download.py         # Auto-download model weights
├── models/
│   ├── nafnet.py           # ConditionalNAFNet (Refusion-HDR)
│   ├── sde.py              # IRSDE diffusion engine
│   └── pu21.py             # PU21 perceptual encoder/decoder
├── tests/                  # 41 tests
└── weights/                # Auto-downloaded, gitignored
```

## License

[MIT](LICENSE)

## Acknowledgments

- [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) — ConditionalNAFNet + IRSDE architecture (AIM 2025 ITM Challenge)
- [PU21](https://doi.org/10.1109/TIP.2021.3070413) — Perceptually Uniform encoding (Mantiuk & Azimi 2021)
- Plate calibration from ARRI ALEXA 35 ACEScg production footage
