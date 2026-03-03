"""
SDR-to-HDR ACEScg Converter

Converts sRGB images to ACEScg half-float EXR using Refusion-HDR
(ConditionalNAFNet + IRSDE diffusion).

Usage:
    python convert.py input.png -o output.exr
    python convert.py ./inputs/ -o ./outputs/
    python convert.py input.png --normalize middle-gray --tile-size 1024
"""
import argparse
import sys
from pathlib import Path

import torch

from pipeline.inference import load_model, convert_file, load_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "lastest_EMA.pth"


def main():
    parser = argparse.ArgumentParser(
        description="Convert sRGB images to ACEScg HDR EXR files."
    )
    parser.add_argument("input", type=str, help="Input image or folder of images")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output EXR path or folder. Default: <input>_acescg.exr")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to lastest_EMA.pth checkpoint")
    parser.add_argument("--normalize", type=str, default="diffuse",
                        choices=["diffuse", "pq-peak", "middle-gray"],
                        help="Luminance normalization mode (default: diffuse)")
    parser.add_argument("--tile-size", type=int, default=2048,
                        help="Max tile dimension in pixels (default: 2048)")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Tile overlap in pixels (default: 64)")
    args = parser.parse_args()

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

    # Load model
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: Weights not found at {weights}")
        print(f"Download from: https://www.dropbox.com/scl/fi/yg44t2i9tgrlsn3c1punc/lastest_EMA.pth?rlkey=fhjb37o34i9yt12337pyed5gi&st=43psqej3&dl=0")
        print(f"Place in: {DEFAULT_WEIGHTS}")
        sys.exit(1)

    print(f"Loading model from {weights}...")
    model = load_model(str(weights), device)
    print(f"Model loaded. Normalization: {args.normalize}")

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
            model=model,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            normalize_mode=args.normalize,
        )

    print(f"\nDone. Processed {len(files)} image(s).")


if __name__ == "__main__":
    main()
