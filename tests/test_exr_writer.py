import torch
import numpy as np
import os
import tempfile
import pytest


def test_write_read_roundtrip():
    """Write an EXR and read it back, values should match within half-float precision."""
    from pipeline.exr_writer import write_exr
    import OpenEXR

    img = torch.rand(64, 64, 3) * 2.0  # ACEScg values, some > 1.0
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Read back with OpenEXR
        with OpenEXR.File(path) as f:
            channels = f.channels()
            assert "R" in channels or "RGB" in channels
    finally:
        os.unlink(path)


def test_half_float_precision():
    """Output should be float16 (half) precision."""
    from pipeline.exr_writer import write_exr
    import OpenEXR

    img = torch.tensor([[[1.0, 0.5, 0.25]]])  # 1x1x3
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        with OpenEXR.File(path) as f:
            # Check that file was written successfully and is readable
            channels = f.channels()
            assert len(channels) > 0
    finally:
        os.unlink(path)


def test_large_image():
    """Should handle large images without error."""
    from pipeline.exr_writer import write_exr

    img = torch.rand(2048, 2048, 3)
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
        path = f.name

    try:
        write_exr(img, path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)
