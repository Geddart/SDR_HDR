# SDR-to-HDR ACEScg Converter — Design Document

**Date:** 2026-03-03
**Status:** Approved

## Goal

CLI tool that converts sRGB images (especially AI-generated) to VFX-ready ACEScg half-float EXR files with proper highlight reconstruction, for use as IBL environment maps and texture maps in 3ds Max (V-Ray/Corona).

## Pipeline

```
sRGB PNG/JPG (uint8)
  → cv2.imread → float32 tensor [0,1] on CUDA
  → Tile if > 2048px on any axis (overlap 64px, cosine-blend seams)
  → Per tile: SDE noise → 100-step reverse diffusion (ConditionalNAFNet)
  → Reassemble tiles
  → PU21 decode → linear luminance in nits (BT.2020 primaries)
  → Normalize (configurable, default: 100 nits = 1.0)
  → 3x3 matrix multiply BT.2020 → ACEScg (GPU)
  → .cpu().numpy().astype(float16)
  → OpenEXR write (ZIP compression, R/G/B channels)
```

## Architecture

Based on Refusion-HDR (AIM 2025 ITM Challenge, 3rd place). Model extracted and modernized for PyTorch 2.9+ / Python 3.13.

### Model: ConditionalNAFNet + IRSDE

- **Backbone:** U-Net encoder-decoder with NAFBlocks (depthwise separable conv, SimpleGate activation, simplified channel attention)
- **Time conditioning:** Sinusoidal embeddings → per-block modulation vectors
- **Encoder:** [1, 1, 1, 28] blocks per level, **Decoder:** [1, 1, 1, 1] blocks
- **Feature width:** 64 channels
- **Down/up:** Strided conv / PixelShuffle
- **SDE:** T=100 timesteps, cosine noise schedule, max_sigma=50
- **Weights:** `lastest_EMA.pth` from Dropbox (architecture-compatible state_dict)

### Color Science

**PU21 decode:** Polynomial mapping from perceptual uniform space to absolute luminance (nits).

**BT.2020 → ACEScg combined matrix** (includes D65→D60 chromatic adaptation):
```
[[ 0.61319930,  0.33951244,  0.04728826],
 [ 0.07021272,  0.91635982,  0.01342745],
 [ 0.02061835,  0.10957647,  0.86980518]]
```

Implemented as `torch.einsum("...c,dc->...d", img, M)` on CUDA.

**Normalization options:**
- `diffuse` (default): divide by 100 nits → diffuse white = 1.0, highlights up to ~100.0
- `pq-peak`: divide by 10000 nits → 1.0 = PQ peak
- `middle-gray`: scale so 18 nits → 0.18

### Tiling

- Threshold: 2048px on any axis (configurable via `--tile-size`)
- Overlap: 64px with cosine-blend feathering
- Sequential tile processing to keep VRAM predictable
- Single 2048x2048 tile ≈ 1GB VRAM with model intermediates

### EXR Output

- Pixel type: HALF (float16)
- Compression: ZIP (16-scanline blocks)
- Channels: R, G, B (separate, VFX convention)
- Compatible with V-Ray / Corona in 3ds Max

## CLI Interface

```bash
# Single image
python convert.py input.png -o output.exr

# Batch (folder)
python convert.py ./inputs/ -o ./outputs/

# Default output name (input_acescg.exr next to source)
python convert.py input.png

# Options
python convert.py input.png --normalize diffuse|pq-peak|middle-gray
python convert.py input.png --tile-size 1024
```

## Project Structure

```
SDR_HDR_Convert/
├── CLAUDE.md
├── convert.py              # CLI entry point
├── models/
│   ├── nafnet.py           # ConditionalNAFNet (extracted, modernized)
│   ├── sde.py              # IRSDE forward/reverse process
│   └── pu21.py             # PU21 encode/decode
├── pipeline/
│   ├── inference.py        # Tiled inference orchestration
│   ├── colorspace.py       # BT.2020 → ACEScg GPU math
│   └── exr_writer.py       # Half-float EXR via OpenEXR
├── weights/
│   └── lastest_EMA.pth     # Pretrained checkpoint (gitignored)
└── docs/
    └── plans/
```

## Dependencies

```
torch>=2.0
torchvision>=0.15
opencv-python
numpy
OpenEXR
einops
timm
tqdm
```

## Modernization from Refusion-HDR

**Extracted as-is (clean PyTorch ops):**
- ConditionalNAFNet architecture (Conv2d, LayerNorm, AdaptiveAvgPool, PixelShuffle)
- IRSDE noise schedule and reverse sampling (tensor arithmetic)

**Rewritten from scratch:**
- Image loading (replaced `torch.ByteStorage.from_buffer()` with cv2 → torch)
- Output pipeline (replaced `.npy` dump with color conversion + EXR)
- Inference orchestration (added tiling, CLI, batch processing)
- PU21 codec (ported from numpy to PyTorch tensors for GPU execution)

**Validation:** Compare output against stock Refusion-HDR on test image to confirm numerical equivalence.

## Use Cases

- **IBL environment maps:** Highlights 50-500x brighter than diffuse surfaces drive realistic lighting
- **Texture maps:** Diffuse white at 1.0 in ACEScg, correct for renderer linear workflow
- **Target DCC:** 3ds Max with V-Ray or Corona, OCIO or manual linear colorspace assignment
