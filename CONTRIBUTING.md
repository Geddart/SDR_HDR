# Contributing

Thanks for your interest in improving SDR→HDR. This is a small, focused tool — contributions that keep it sharp and correct are very welcome.

## Getting set up

```bash
git clone https://github.com/Geddart/SDR_HDR.git
cd SDR_HDR
pip install -e ".[dev]"
```

A CUDA GPU is only needed for `--mode model`. Everything else (linear mode, the full test suite) runs on CPU.

## Running the tests

```bash
pytest -m "not network" -q      # the suite CI runs (no network, GPU tests auto-skip)
pytest                          # everything, including the live weights-URL check
```

All tests must pass before a PR can merge. CI runs them on Python 3.10–3.12.

## Guidelines

- **Color science is verified — don't guess.** The `REC709_TO_ACESCG` matrix and the inverse-tonemap curve are validated by a synthetic round-trip test. If you change color math, add a numerical test that proves it (see `tests/test_colorspace.py` and `TestRoundTrip` in `tests/test_inference.py`).
- **Don't rename model layers or change tensor shapes** in `models/`. The pretrained weights load by exact `state_dict` match; renames silently break loading.
- **Keep `convert.py` thin.** It's the CLI entry point — put real logic in `pipeline/`.
- Match the surrounding style. Add a test with any behavioral change.
- One logical change per PR; a clear description of *why* beats a long description of *what*.

## Reporting bugs / requesting features

Use the issue templates. For conversion problems, include the input image characteristics, the exact command, and what you expected vs. got — colorimetric bugs are impossible to diagnose without numbers.

## Third-party code

This project reuses permissively-licensed model code (see `THIRD_PARTY_LICENSES.md`). If you touch `models/`, preserve the upstream attribution headers.
