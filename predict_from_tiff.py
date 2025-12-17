import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

import numpy as np
import tifffile as tiff


# DeepWaterMap expects: Blue, Green, Red, NIR, SWIR1, SWIR2
# For Sentinel-2 -> B2, B3, B4, B8, B11, B12
DWM_REQUIRED_NAMES = ["B2","B3","B4","B8","B11","B12"]

# Αν το TIFF σου έχει 12 bands σε σειρά:
# [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]
# τότε οι 1-based δείκτες για (B2,B3,B4,B8,B11,B12) είναι:
DEFAULT_BAND_IDX_1BASED = [2, 3, 4, 8, 11, 12]


def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Accepts (H,W,C) or (C,H,W) and returns (H,W,C)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array from TIFF, got {arr.shape}")
    # if first dim looks like channels
    if arr.shape[0] in (2, 6, 10, 12, 13) and arr.shape[1] > 16 and arr.shape[2] > 16:
        return np.transpose(arr, (1, 2, 0))
    return arr


def normalize_s2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.nanmax(x))
    if 1.5 < mx <= 20000:
        x = x / 10000.0
    return np.clip(x, 0.0, 1.0)


def select_bands_by_idx(hwc: np.ndarray, idx_1based: List[int]) -> np.ndarray:
    c = hwc.shape[2]
    idx0 = [i - 1 for i in idx_1based]
    if any(i < 0 or i >= c for i in idx0):
        raise ValueError(f"band_idx out of range for C={c}. Got 1-based={idx_1based}")
    out = np.stack([hwc[:, :, i] for i in idx0], axis=2)
    if out.shape[2] != 6:
        raise ValueError("Expected 6 selected bands.")
    return out


def run_inference(inference_py: str, checkpoint_path: str, image_tif_path: str, out_png: str) -> None:
    cmd = [
        sys.executable, inference_py,
        "--checkpoint_path", checkpoint_path,
        "--image_path", image_tif_path,
        "--save_path", out_png,
    ]
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "DeepWaterMap inference failed.\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tif", required=True, help="Input TIFF (Sentinel-2 multiband or already-6band)")
    ap.add_argument("--out_png", required=True, help="Output PNG path")
    ap.add_argument("--checkpoint_path", default="checkpoints/cp.135.ckpt")
    ap.add_argument("--inference_py", default="inference.py")
    ap.add_argument(
        "--band_idx",
        default=",".join(map(str, DEFAULT_BAND_IDX_1BASED)),
        help="1-based indices to pick bands for DWM (default assumes 12-band order B1..B12 with B8A,B9,B11,B12). "
             "Default: 2,3,4,8,11,12",
    )
    ap.add_argument("--already_6band", action="store_true",
                    help="Set this if the input TIFF is already 6-band in DWM order (B2,B3,B4,B8,B11,B12).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".", exist_ok=True)

    if args.already_6band:
        # run directly
        run_inference(args.inference_py, args.checkpoint_path, args.in_tif, args.out_png)
        print("Done:", args.out_png)
        return

    band_idx = [int(x.strip()) for x in args.band_idx.split(",") if x.strip()]

    # read TIFF -> select -> temp 6-band tif -> inference
    arr = tiff.imread(args.in_tif)
    hwc = to_hwc(arr)
    hwc = normalize_s2(hwc)
    hwc6 = select_bands_by_idx(hwc, band_idx)

    with tempfile.TemporaryDirectory() as td:
        tmp6 = os.path.join(td, "tmp_s2_6band.tif")
        tiff.imwrite(tmp6, hwc6.astype(np.float32), dtype=np.float32)
        run_inference(args.inference_py, args.checkpoint_path, tmp6, args.out_png)

    print("Done:", args.out_png)


if __name__ == "__main__":
    main()
