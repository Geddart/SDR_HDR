# SDR-to-HDR ACEScg Converter

## Project Overview

CLI tool converting sRGB images to ACEScg half-float EXR with AI-based highlight reconstruction. Uses extracted/modernized Refusion-HDR model (ConditionalNAFNet + IRSDE diffusion). Primary use: converting AI-generated images to VFX-ready HDR assets for 3ds Max lighting and rendering.

## Tech Stack

- Python 3.13, PyTorch 2.9+ (CUDA 12.8)
- Target GPU: RTX 5090 (32GB)
- OpenEXR for half-float EXR output
- No oiiotool/OCIO dependency — color math is pure PyTorch on GPU

## Key Architecture Decisions

- Model code extracted from Refusion-HDR and modernized for PyTorch 2.x (removed deprecated ByteStorage, ByteTensor APIs)
- Pretrained weights loaded via standard state_dict (architecture must match exactly)
- Color conversion (Rec.709 → ACEScg) done as GPU 3x3 matrix multiply with Bradford D65→D60, not OCIO
- PU21 decode runs on GPU tensors, not numpy
- Tiling with cosine-blend overlap for images > 2048px
- Both linear and model modes share the same inverse tonemap curve: `L * (gain + (peak-gain) * L^power)`
- Scene presets (night, day, interior, overcast, hdri, roundtrip) set peak/gain/power/dither defaults

## File Conventions

- `models/` — Neural network architecture (NAFNet, SDE, PU21). Do not modify layer names/shapes or weights won't load.
- `pipeline/` — Inference orchestration, color science, EXR output.
- `convert.py` — CLI entry point. Keep thin, delegate to pipeline/.
- `weights/` — Gitignored. Contains `lastest_EMA.pth` (note: upstream typo in filename, keep as-is).

## Color Science Reference

- Input: sRGB (Rec.709 primaries, sRGB transfer function)
- Model output: PU21-encoded, decoded to Rec.709 luminance in nits (NOT BT.2020)
- Final output: ACEScg (AP1 primaries, D60 white point, scene-linear)
- Rec.709→ACEScg matrix includes Bradford D65→D60 chromatic adaptation
- Inverse tonemap: `L * (gain + (peak-gain) * L^power)` — applied to luminance only, RGB scaled proportionally
- TPDF dithering at ±1 LSB before nonlinear expansion (disable with `--no-dither` for round-trips)
- The REC709_TO_ACESCG matrix is verified correct — never blend or scale it

## Testing

- 53 tests covering colorspace, inference, presets, power, dithering, and round-trip
- Synthetic round-trip test: ACEScg → sRGB 8-bit → ACEScg with <1% per-channel error
- Check EXR readability in Nuke/3ds Max
- Verify ACEScg values with colour-science library spot checks
