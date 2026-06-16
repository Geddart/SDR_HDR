# Security Policy

## Supported versions

The latest release on the `master` branch is the only supported version. Fixes are
applied to `master` and shipped in the next tagged release.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/Geddart/SDR_HDR/security/advisories/new), or
- email **sascha@geddart.de** with a description and reproduction steps.

You can expect an initial response within a few days. Please give a reasonable
window to address the issue before any public disclosure.

## Scope notes

This is an offline image-conversion CLI. The most relevant risks are:

- **Model weights** are downloaded over HTTPS from this repository's GitHub Releases
  and validated by size (see `pipeline/download.py`). They are not integrity-checked
  by hash — if you need supply-chain assurance, download and verify the weights
  manually and pass them with `--weights`.
- Image decoding relies on OpenCV/OpenEXR; untrusted input images are parsed by
  those libraries.
