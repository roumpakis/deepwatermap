#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from typing import List

try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Missing dependency: tifffile. Install with: pip install tifffile") from e


# Default assumption για Sentinel-2 12-band npy:
# [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]
DEFAULT_S2_12_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]

# DeepWaterMap expects (approx Landsat) bands:
# Blue, Green, Red, NIR, SWIR1, SWIR2  -> for Sentinel-2: B2,B3,B4,B8,B11,B12
DWM_REQUIRED = ["B2", "B3", "B4", "B8", "B11", "B12"]


def load_npy_any_layout(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")
    # accept (C,H,W) or (H,W,C)
    if arr.shape[0] in (2, 6, 10, 12, 13) and arr.shape[1] > 16 and arr.shape[2] > 16:
        # (C,H,W) -> (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def normalize_s2(x: np.ndarray) -> np.ndarray:
    """Heuristic normalization:
    - if values look like 0..10000 (common for reflectance-scaled ints), divide by 10000
    - else keep as-is
    Then clip to [0,1] lightly.
    """
    x = x.astype(np.float32)
    mx = float(np.nanmax(x))
    if mx > 1.5 and mx <= 20000:
        x = x / 10000.0
    # mild clipping
    x = np.clip(x, 0.0, 1.0)
    return x


#def select_bands(s2_hwc: np.ndarray, band_names: List[str]) -> np.ndarray:
def select_bands(s2_hwc, band_names):

    if s2_hwc.shape[2] != len(band_names):
        raise ValueError(f"Band name list length {len(band_names)} != channels {s2_hwc.shape[2]}")
    idx = {b: i for i, b in enumerate(band_names)}
    missing = [b for b in DWM_REQUIRED if b not in idx]
    if missing:
        raise ValueError(f"Missing required bands for DeepWaterMap: {missing}. "
                         f"Available={band_names}")
    out = np.stack([s2_hwc[:, :, idx[b]] for b in DWM_REQUIRED], axis=2)  # (H,W,6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_npy", required=True, help="Path to Sentinel-2 npy (C,H,W) or (H,W,C)")
    ap.add_argument("--checkpoint_path", required=True, help="DeepWaterMap checkpoint, e.g. checkpoints/cp.135.ckpt")
    ap.add_argument("--out_png", required=True, help="Output PNG path for water map")
    ap.add_argument("--band_order", default=",".join(DEFAULT_S2_12_BANDS),
                    help="Comma-separated band names matching your channels (default assumes 12-band S2: "
                         + ",".join(DEFAULT_S2_12_BANDS) + ")")
    ap.add_argument("--inference_py", default="inference.py", help="Path to DeepWaterMap inference.py (default: inference.py)")
    args = ap.parse_args()

    band_names = [b.strip() for b in args.band_order.split(",") if b.strip()]

    s2 = load_npy_any_layout(args.s2_npy)  # (H,W,C)
    s2 = normalize_s2(s2)

    s2_6 = select_bands(s2, band_names)  # (H,W,6)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_tif = os.path.join(td, "tmp_s2_6band.tif")
        # γράφουμε 6-band TIFF (H,W,6)
        tiff.imwrite(tmp_tif, s2_6, dtype=np.float32)

        cmd = [
            sys.executable, args.inference_py,
            "--checkpoint_path", args.checkpoint_path,
            "--image_path", tmp_tif,
            "--save_path", args.out_png,
        ]
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise SystemExit(f"DeepWaterMap inference failed (exit code {res.returncode}).")

    print("Done:", args.out_png)


if __name__ == "__main__":
    main()
