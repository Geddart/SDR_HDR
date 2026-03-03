"""
Automatic model weight download from GitHub Releases.
"""
import os
import sys
import urllib.request

WEIGHTS_URL = "https://github.com/Geddart/SDR_HDR/releases/download/v1.0.0/lastest_EMA.pth"
EXPECTED_SIZE_MB = 292
SIZE_TOLERANCE_MB = 20


def _progress_hook(block_num, block_size, total_size):
    """Print download progress on a single line."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_done:.0f}/{mb_total:.0f} MB ({pct}%)")
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_done:.0f} MB")
    sys.stdout.flush()


def download_weights(dest_path: str, max_retries: int = 1) -> None:
    """
    Download model weights to dest_path.

    Downloads to a .tmp file first and renames on success to avoid partial files.
    Verifies file size is approximately correct (~292 MB).
    """
    tmp_path = dest_path + ".tmp"
    print(f"  Model weights not found. Downloading (~{EXPECTED_SIZE_MB} MB)...")

    for attempt in range(max_retries + 1):
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, tmp_path, reporthook=_progress_hook)
            print()  # newline after progress

            # Verify size
            actual_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if abs(actual_size_mb - EXPECTED_SIZE_MB) > SIZE_TOLERANCE_MB:
                os.remove(tmp_path)
                raise RuntimeError(
                    f"Downloaded file is {actual_size_mb:.0f} MB, expected ~{EXPECTED_SIZE_MB} MB. "
                    f"The download may be corrupted or the URL may have changed."
                )

            # Atomic rename
            os.replace(tmp_path, dest_path)
            print(f"  Weights saved to {dest_path}")
            return

        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < max_retries:
                print(f"\n  Download failed ({e}), retrying...")
            else:
                print(f"\n  Error: Failed to download model weights: {e}")
                print(f"  Manual download: {WEIGHTS_URL}")
                print(f"  Place the file at: {dest_path}")
                sys.exit(1)
