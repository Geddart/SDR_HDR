# SDR to HDR — ACEScg Converter

Turn any 8-bit image into a production-ready ACEScg EXR. AI-reconstructed highlights, physically plausible dynamic range, ready to light a scene.

```bash
pip install .
sdr-hdr photo.png --mode model -o photo_acescg.exr
```

Weights download automatically. Tile size adapts to your GPU. Output is half-float RGBA EXR with premultiplied alpha — drops straight into Nuke, RV, 3ds Max, Houdini, or any ACES-aware app.

## Install

```bash
git clone https://github.com/Geddart/SDR_HDR.git
cd SDR_HDR
pip install .
```

Python 3.10+, PyTorch 2.0+. CUDA GPU with 8+ GB VRAM for model mode.

## Usage

```bash
# AI highlight reconstruction (recommended — needs GPU)
sdr-hdr photo.png --mode model -o photo_acescg.exr

# Fast math-only conversion (no GPU needed)
sdr-hdr photo.png -o photo_acescg.exr

# Batch convert a folder
sdr-hdr ./textures/ -o ./textures_hdr/
```

## Scene Presets

Presets are tuned for common scene types. Pick the one that matches your source and go.

```bash
sdr-hdr photo.png --preset day -o photo_acescg.exr
sdr-hdr photo.png --preset night -o photo_acescg.exr
sdr-hdr photo.png --preset hdri -o photo_acescg.exr

# Round-trip: restore ACEScg data that was saved as sRGB PNG
sdr-hdr texture.png --preset roundtrip -o texture_acescg.exr
```

| Preset | Peak | Gain | Power | What it's for |
|--------|------|------|-------|---------------|
| `night` | 150 | 3.0 | 2 | Night exteriors — point lights very hot, deep shadows |
| `day` | 30 | 4.0 | 2 | Daylight — lower contrast, ambient-filled shadows |
| `interior` | 60 | 3.5 | 2 | Indoor scenes — windows and practicals moderately hot |
| `overcast` | 15 | 5.0 | 2 | Flat lighting — gentle expansion, lifted shadows |
| `hdri` | 250 | 3.0 | 2 | Environment maps for CG lighting — very hot light sources |
| `roundtrip` | 1 | 1.0 | 50 | Restore ACEScg data saved as sRGB PNG — no expansion, no dither |

`--peak`, `--gain`, and `--power` override preset values if you need to fine-tune:

```bash
# Start with night preset but cap highlights lower
sdr-hdr photo.png --preset night --peak 100
```

Without a preset, defaults are peak=150, gain=3.0, power=2.

## Two Modes

### Model Mode (`--mode model`)

A [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) diffusion model (ConditionalNAFNet + IRSDE) runs 100 reverse diffusion steps to reconstruct what was behind clipped highlights. PU21 decodes to absolute luminance, then the same luminance-preserving inverse tonemap as linear mode expands the dynamic range using `--peak` and `--gain`. Both modes produce matching output ranges — the model contributes smarter highlight reconstruction.

Weights (~292 MB) auto-download on first run. Tile size auto-adapts to your VRAM.

### Linear Mode (default)

Pure math, no AI. Fast, runs on CPU.

```
sRGB → EOTF decode → TPDF dither → luminance-based inverse tonemap → Rec.709 → ACEScg
```

Applies `hdr = Y * (gain + (peak - gain) * Y^power)` to luminance only, scales RGB proportionally. Preserves chromaticity, prevents solarization. `--power` controls how steeply highlights ramp up: low power (2) spreads expansion across the range, high power (50) concentrates it near white. TPDF dithering kills 8-bit banding before the nonlinear expansion; disable with `--no-dither` for clean round-trips.

## Parameters

```
sdr-hdr input -o output [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `linear` | `linear` or `model` |
| `--preset` | — | Scene preset: `night`, `day`, `interior`, `overcast`, `hdri`, `roundtrip` |
| `--exposure` | `0.0` | Exposure adjustment in stops |
| `--peak` | `150` | Max HDR value — or set by preset |
| `--gain` | `3.0` | Mid-tone multiplier — or set by preset |
| `--tile-size` | auto | Tile dimension, auto from VRAM (model mode) |
| `--overlap` | `64` | Tile overlap for blending (model mode) |
| `--power` | `2` | Highlight rolloff steepness — or set by preset |
| `--no-dither` | off | Disable TPDF dithering (cleaner for round-trips) |
| `--weights` | auto | Path to model checkpoint (model mode) |

## How It Works

All color math is pure PyTorch on GPU — no OCIO, no oiiotool.

- **Input**: sRGB (Rec.709 primaries, sRGB EOTF)
- **Output**: ACEScg (AP1 primaries, D60 white point, scene-linear)
- **Color transform**: 3x3 matrix with Bradford chromatic adaptation D65 → D60
- **Dithering**: TPDF at ±1 LSB before nonlinear expansion
- **Alpha**: Premultiplied, preserved from source

## Project Structure

```
SDR_HDR/
├── convert.py              # CLI entry point + presets
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
├── tests/                  # 53 tests
└── weights/                # Auto-downloaded, gitignored
```

## License

[MIT](LICENSE)

## Acknowledgments

- [Refusion-HDR](https://github.com/LiamLian0727/Refusion-HDR) — ConditionalNAFNet + IRSDE architecture (AIM 2025 ITM Challenge)
- [PU21](https://doi.org/10.1109/TIP.2021.3070413) — Perceptually Uniform encoding (Mantiuk & Azimi 2021)
