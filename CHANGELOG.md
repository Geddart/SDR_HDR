# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-05

### Added
- `--power` parameter controlling highlight rolloff steepness in the inverse tonemap.
- `roundtrip` preset for restoring ACEScg data saved as sRGB PNG (no expansion, no dither).
- `--no-dither` flag to disable TPDF dithering for clean round-trips.

### Changed
- Dithering is now conditional and documented alongside the other expansion controls.

## [1.2.0] - 2026-03-03

### Added
- Scene presets: `night`, `day`, `interior`, `overcast`, `hdri`, each tuning
  peak / gain / power for a common scene type.

## [1.1.0] - 2026-03-03

### Added
- Automatic VRAM-based tile sizing for model mode.
- Automatic model-weight download from GitHub Releases on first run.
- `pip install .` support via `pyproject.toml`.

## [1.0.0] - 2026-03-03

### Added
- Initial release: sRGB → ACEScg half-float EXR conversion.
- Linear mode (pure-PyTorch color math) and model mode (ConditionalNAFNet + IRSDE
  highlight reconstruction extracted from Refusion-HDR).
- Luminance-based inverse tonemap, PU21 decode, Rec.709→ACEScg with Bradford D65→D60.
- Half-float RGBA EXR output with premultiplied alpha.

[1.3.0]: https://github.com/Geddart/SDR_HDR/releases/tag/v1.3.0
[1.2.0]: https://github.com/Geddart/SDR_HDR/releases/tag/v1.2.0
[1.1.0]: https://github.com/Geddart/SDR_HDR/releases/tag/v1.1.0
[1.0.0]: https://github.com/Geddart/SDR_HDR/releases/tag/v1.0.0
