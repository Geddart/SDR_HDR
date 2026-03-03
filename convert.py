"""
SDR-to-HDR ACEScg Converter

Converts sRGB images to ACEScg half-float EXR with HDR highlight expansion.

Modes:
  linear  — sRGB EOTF -> inverse tone map -> Rec.709 -> ACEScg (fast, predictable)
  model   — AI-based highlight reconstruction via Refusion-HDR diffusion

Usage:
    python convert.py input.png -o output.exr
    python convert.py ./inputs/ -o ./outputs/
    python convert.py input.png --peak 500
    python convert.py input.png --mode model --exposure -4
"""
import argparse
import os
import sys
from pathlib import Path

import torch

from pipeline.inference import load_model, convert_file, estimate_tile_size


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
WEIGHTS_FILENAME = "lastest_EMA.pth"

PRESETS = {
    "night":    {"peak": 150.0, "gain": 3.0},
    "day":      {"peak": 30.0,  "gain": 4.0},
    "interior": {"peak": 60.0,  "gain": 3.5},
    "overcast": {"peak": 15.0,  "gain": 5.0},
    "hdri":     {"peak": 250.0, "gain": 3.0},
}


def _resolve_weights_path() -> Path:
    """Find weights: project-relative first, then user cache dir."""
    # 1. Project-relative (git clone + python convert.py)
    project_weights = Path(__file__).parent / "weights" / WEIGHTS_FILENAME
    if project_weights.exists():
        return project_weights

    # 2. User cache dir (pip install)
    if sys.platform == "win32":
        cache_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "sdr-hdr"
    else:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sdr-hdr"

    cache_weights = cache_dir / WEIGHTS_FILENAME
    if cache_weights.exists():
        return cache_weights

    # Neither exists — return project-relative if weights/ dir exists, else cache
    if (Path(__file__).parent / "weights").is_dir():
        return project_weights
    return cache_weights


DEFAULT_WEIGHTS = _resolve_weights_path()


def main():
    parser = argparse.ArgumentParser(
        description="Convert sRGB images to ACEScg HDR EXR files."
    )
    parser.add_argument("input", type=str, help="Input image or folder of images")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output EXR path or folder. Default: <input>_acescg.exr")
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "model"],
                        help="Conversion mode (default: linear)")
    parser.add_argument("--exposure", type=float, default=0.0,
                        help="Exposure adjustment in stops (default: 0.0)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to lastest_EMA.pth checkpoint (model mode only)")
    parser.add_argument("--normalize", type=str, default="diffuse",
                        choices=["diffuse", "pq-peak", "middle-gray"],
                        help="Luminance normalization (model mode only, default: diffuse)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Max tile dimension in pixels (model mode only, default: auto based on VRAM)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Tile overlap in pixels (model mode only, default: 64)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Scene preset: night, day, interior, overcast, hdri (sets peak and gain)")
    parser.add_argument("--peak", type=float, default=None,
                        help="Peak HDR value for sRGB white (linear mode, default: 150 or from preset)")
    parser.add_argument("--gain", type=float, default=None,
                        help="Mid-tone brightness multiplier (linear mode, default: 3.0 or from preset)")
    args = parser.parse_args()

    # Resolve peak/gain: preset -> explicit override -> fallback defaults
    peak = args.peak
    gain = args.gain
    if args.preset:
        preset_vals = PRESETS[args.preset]
        if peak is None:
            peak = preset_vals["peak"]
        if gain is None:
            gain = preset_vals["gain"]
    if peak is None:
        peak = 150.0
    if gain is None:
        gain = 3.0

    # Resolve input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Collect files
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            print(f"Error: No images found in {input_path}")
            sys.exit(1)
    else:
        files = [input_path]

    # Resolve output
    if args.output:
        output_path = Path(args.output)
        if len(files) > 1:
            output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Exposure: {args.exposure:+.1f} EV")
    if args.mode == "linear":
        if args.preset:
            print(f"Preset: {args.preset} (peak={peak:.0f}, gain={gain:.1f})")
        else:
            print(f"Peak: {peak:.0f}, Gain: {gain:.1f}")
    elif args.preset:
        print(f"Note: --preset is ignored in model mode")

    # Load model (only if needed)
    model = None
    tile_size = args.tile_size
    if args.mode == "model":
        weights = Path(args.weights)
        if not weights.exists():
            # Auto-download only for default path; custom --weights is a user error
            if args.weights != str(DEFAULT_WEIGHTS):
                print(f"Error: Weights not found at {weights}")
                sys.exit(1)
            from pipeline.download import download_weights
            weights.parent.mkdir(parents=True, exist_ok=True)
            download_weights(str(weights))
        print(f"Loading model from {weights}...")
        model = load_model(str(weights), device)
        print(f"Model loaded. Normalization: {args.normalize}")

        # Auto tile-size based on available VRAM
        if tile_size is None:
            tile_size = estimate_tile_size(device)
        print(f"Tile size: {tile_size}px")

    # Process
    for i, file in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {file.name}")

        if output_path is None:
            out = file.with_name(file.stem + "_acescg.exr")
        elif output_path.suffix == ".exr":
            out = output_path
        else:
            out = output_path / (file.stem + "_acescg.exr")

        convert_file(
            input_path=str(file),
            output_path=str(out),
            device=device,
            mode=args.mode,
            model=model,
            tile_size=tile_size or 1024,
            overlap=args.overlap,
            normalize_mode=args.normalize,
            exposure=args.exposure,
            peak=peak,
            gain=gain,
        )

    print(f"\nDone. Processed {len(files)} image(s).")


if __name__ == "__main__":
    main()
