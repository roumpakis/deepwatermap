import argparse
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt


WINDOW_RE = re.compile(r"^\d{8}_\d{8}$")

# Assumption for S2 12-band npy order:
# [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]
DEFAULT_S2_12_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]

# DeepWaterMap expects: Blue,Green,Red,NIR,SWIR1,SWIR2
# Sentinel-2 mapping: B2,B3,B4,B8,B11,B12
DWM_REQUIRED = ["B2","B3","B4","B8","B11","B12"]


# -----------------------
# IO / helpers
# -----------------------
def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Accepts (H,W,C) or (C,H,W) and returns (H,W,C)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[0] in (2, 6, 10, 12, 13) and arr.shape[1] > 16 and arr.shape[2] > 16:
        return np.transpose(arr, (1, 2, 0))
    return arr


def normalize_s2(x: np.ndarray) -> np.ndarray:
    """
    Heuristic normalization:
    - if max looks like 0..10000 (common scaled reflectance), divide by 10000
    - clip to [0,1]
    """
    x = x.astype(np.float32)
    mx = float(np.nanmax(x))
    if 1.5 < mx <= 20000:
        x = x / 10000.0
    return np.clip(x, 0.0, 1.0)


def select_bands_for_dwm(hwc: np.ndarray, band_names: List[str]) -> np.ndarray:
    if hwc.shape[2] != len(band_names):
        raise ValueError(f"band_order length {len(band_names)} != channels {hwc.shape[2]}")
    idx = {b: i for i, b in enumerate(band_names)}
    missing = [b for b in DWM_REQUIRED if b not in idx]
    if missing:
        raise ValueError(f"Missing required bands for DWM: {missing}. Available={band_names}")
    return np.stack([hwc[:, :, idx[b]] for b in DWM_REQUIRED], axis=2)  # (H,W,6)


def compute_ndwi(hwc: np.ndarray, band_names: List[str], thr: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    idx = {b: i for i, b in enumerate(band_names)}
    if "B3" not in idx or "B8" not in idx:
        raise ValueError("NDWI needs B3 (Green) and B8 (NIR). Check --band_order.")
    g = hwc[:, :, idx["B3"]].astype(np.float32)
    n = hwc[:, :, idx["B8"]].astype(np.float32)
    ndwi = (g - n) / (g + n + 1e-6)
    mask = ndwi > float(thr)
    return ndwi, mask


def run_dwm(inference_py: str, checkpoint_path: str, image_tif_path: str, out_png_path: str) -> None:
    cmd = [
        sys.executable, inference_py,
        "--checkpoint_path", checkpoint_path,
        "--image_path", image_tif_path,
        "--save_path", out_png_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "DeepWaterMap inference failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def load_pred_png_prob01(png_path: str) -> np.ndarray:
    """
    Loads model output PNG (grayscale) and returns a float map in [0,1].
    If PNG is binary, this will just be 0/1.
    """
    im = Image.open(png_path).convert("L")
    arr = np.array(im).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_prob_tiff(prob01: np.ndarray, path: Path) -> None:
    """Saves float32 TIFF with values in [0,1]."""
    prob01 = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    tiff.imwrite(str(path), prob01, dtype=np.float32)


def save_binary_png(mask: np.ndarray, path: Path) -> None:
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(str(path))


def save_prob_bluewhite_png(prob01: np.ndarray, path: Path) -> None:
    """
    Blue-White color mask:
      p=0  -> white (255,255,255)
      p=1  -> blue  (0,0,255)
    Interpolation: rgb = (1-p)*white + p*blue => [1-p, 1-p, 1]
    """
    p = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    r = (1.0 - p)
    g = (1.0 - p)
    b = np.ones_like(p)
    rgb = np.stack([r, g, b], axis=2)
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(str(path))


def save_overlay_on_rgb(base_rgb01: np.ndarray, prob01: np.ndarray, path: Path, max_alpha: float = 0.65) -> None:
    """
    Overlay probability as blue tint on RGB:
      alpha = max_alpha * prob
      overlay color = pure blue
    """
    base = np.clip(base_rgb01.astype(np.float32), 0.0, 1.0)
    p = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    overlay = np.zeros_like(base)
    overlay[..., 2] = 1.0  # blue
    alpha = (max_alpha * p)[..., None]
    out = base * (1.0 - alpha) + overlay * alpha
    out = np.clip(out, 0.0, 1.0)
    Image.fromarray((out * 255).astype(np.uint8), mode="RGB").save(str(path))


def quick_rgb(hwc: np.ndarray, band_names: List[str]) -> Optional[np.ndarray]:
    idx = {b: i for i, b in enumerate(band_names)}
    for b in ("B2","B3","B4"):
        if b not in idx:
            return None
    r = hwc[:, :, idx["B4"]]
    g = hwc[:, :, idx["B3"]]
    b = hwc[:, :, idx["B2"]]
    rgb = np.stack([r, g, b], axis=2).astype(np.float32)
    p2 = np.nanpercentile(rgb, 2)
    p98 = np.nanpercentile(rgb, 98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    return np.clip(rgb, 0, 1)


# -----------------------
# Metrics
# -----------------------
def metrics_binary(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch pred={pred.shape}, ref={ref.shape}")

    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    total = tp + fp + fn + tn

    iou = tp / (tp + fp + fn + 1e-9)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    acc = (tp + tn) / (total + 1e-9)
    diff_pct = (fp + fn) / (total + 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
        "iou": iou, "f1": f1, "precision": precision, "recall": recall,
        "accuracy": acc, "diff_pct": diff_pct,
        "pred_water_pct": pred.sum() / (total + 1e-9),
        "ndwi_water_pct": ref.sum() / (total + 1e-9),
        "fp_pct": fp / (total + 1e-9),
        "fn_pct": fn / (total + 1e-9),
    }


def sweep_thresholds(pred_prob01: np.ndarray, ref_mask: np.ndarray, step: float = 0.01):
    thrs = np.arange(0.0, 1.0 + 1e-9, step)
    rows = []
    best = None
    for t in thrs:
        pred = pred_prob01 >= t
        m = metrics_binary(pred, ref_mask)
        rows.append({"thr": float(t), **m})
        if best is None or m["f1"] > best[1]["f1"]:
            best = (float(t), m)
    curve = pd.DataFrame(rows)
    return best[0], best[1], curve


# -----------------------
# Window discovery
# -----------------------
def find_windows(data_root: Path):
    hits = []
    for p in data_root.rglob("s2_image.npy"):
        window = None
        for part in p.parts:
            if WINDOW_RE.match(part):
                window = part
                break
        if window is None:
            window = p.parent.parent.name  # fallback

        hits.append({
            "window": window,
            "sentinel_dir": str(p.parent),
            "s2_path": str(p),
        })

    def keyfn(x):
        try:
            start = x["window"].split("_")[0]
            return datetime.strptime(start, "%Y%m%d")
        except Exception:
            return datetime.max

    hits.sort(key=keyfn)
    return hits


# -----------------------
# Plotting
# -----------------------
def make_plots(
    window_out: Path,
    window: str,
    rgb01: Optional[np.ndarray],
    ndwi: np.ndarray,
    ndwi_mask: np.ndarray,
    pred_prob01: np.ndarray,
    chosen_thr: float,
    m: dict,
    curve_df: pd.DataFrame,
):
    # Summary 2x2
    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    if rgb01 is not None:
        ax1.imshow(rgb01)
        ax1.set_title(f"{window} - RGB")
    else:
        ax1.imshow(ndwi, cmap="gray")
        ax1.set_title(f"{window} - NDWI (no RGB bands)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(ndwi_mask, cmap="gray")
    ax2.set_title(f"NDWI mask (thr=0.0) water={m['ndwi_water_pct']*100:.2f}%")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(pred_prob01, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("DeepWaterMap score/prob map [0,1] (no threshold)")
    ax3.axis("off")

    pred_bin = pred_prob01 >= chosen_thr

    # diff: 0 TN, 1 TP, 2 FP, 3 FN
    diff = np.zeros_like(pred_bin, dtype=np.uint8)
    diff[np.logical_and(pred_bin, ndwi_mask)] = 1
    diff[np.logical_and(pred_bin, ~ndwi_mask)] = 2
    diff[np.logical_and(~pred_bin, ndwi_mask)] = 3

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(diff, cmap="viridis")
    ax4.set_title(f"Diff @thr={chosen_thr:.2f}  IoU={m['iou']:.3f}  F1={m['f1']:.3f}  diff={m['diff_pct']*100:.2f}%")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(str(window_out / f"{window}_summary.png"), dpi=150)
    plt.close(fig)

    # NDWI histogram
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    vals = ndwi[np.isfinite(ndwi)].ravel()
    ax.hist(vals, bins=80)
    ax.axvline(0.0, linestyle="--")
    ax.set_title(f"{window} NDWI histogram")
    fig.tight_layout()
    fig.savefig(str(window_out / f"{window}_ndwi_hist.png"), dpi=150)
    plt.close(fig)

    # Threshold sweep curve (F1 / IoU)
    if len(curve_df) > 1:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(curve_df["thr"], curve_df["f1"], label="F1")
        ax.plot(curve_df["thr"], curve_df["iou"], label="IoU")
        ax.axvline(chosen_thr, linestyle="--")
        ax.set_title(f"{window} Threshold sweep vs NDWI")
        ax.set_xlabel("threshold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(window_out / f"{window}_thr_sweep.png"), dpi=150)
        plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"D:\models\Faneromeni\eab3606b084b")
    ap.add_argument("--out_root", required=True, help=r"D:\models\Faneromeni\eab3606b084b\outputs_dwm_vs_ndwi")
    ap.add_argument("--checkpoint_path", required=True, help=r"D:\models\deepwatermap\checkpoints\cp.135.ckpt")
    ap.add_argument("--inference_py", default="inference.py")
    ap.add_argument("--band_order", default=",".join(DEFAULT_S2_12_BANDS),
                    help="Comma-separated band names matching channels in s2_image.npy")
    ap.add_argument("--ndwi_thr", type=float, default=0.0)

    # If pred_thr is None -> auto-pick best threshold by F1 vs NDWI (only for metrics/plots)
    ap.add_argument("--pred_thr", type=float, default=None)
    ap.add_argument("--sweep_step", type=float, default=0.01)
    ap.add_argument("--max_pixels", type=int, default=25000000)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    band_names = [b.strip() for b in args.band_order.split(",") if b.strip()]

    windows = find_windows(data_root)
    if not windows:
        raise SystemExit(f"No s2_image.npy found under: {data_root}")

    rows = []

    for item in windows:
        window = item["window"]
        s2_path = Path(item["s2_path"])
        sentinel_dir = Path(item["sentinel_dir"])

        window_out = out_root / window
        window_out.mkdir(parents=True, exist_ok=True)

        print(f"[{window}] {s2_path}")

        # Load S2
        s2 = np.load(str(s2_path), allow_pickle=False)
        s2 = to_hwc(s2)
        h, w, c = s2.shape
        if h * w > args.max_pixels:
            print(f"  SKIP (too large): {h}x{w}")
            continue

        s2n = normalize_s2(s2)

        # NDWI "reference"
        ndwi, ndwi_mask = compute_ndwi(s2n, band_names, thr=args.ndwi_thr)

        # Prepare 6-band tif for DWM
        s2_6 = select_bands_for_dwm(s2n, band_names)

        # Output names (match window)
        pred_score_png = window_out / f"{window}_pred_score.png"        # raw model output PNG
        pred_prob_npy = window_out / f"{window}_pred_prob01.npy"        # float [0,1]
        pred_prob_tif = window_out / f"{window}_pred_prob01.tif"        # float32 Geo-less tif (matrix)
        pred_prob_bluewhite = window_out / f"{window}_pred_bluewhite.png"
        ndwi_mask_png = window_out / f"{window}_ndwi_thr{args.ndwi_thr:.1f}.png"

        # Run DeepWaterMap -> pred_score_png
        with tempfile.TemporaryDirectory() as td:
            tmp6 = Path(td) / "tmp_s2_6band.tif"
            tiff.imwrite(str(tmp6), s2_6.astype(np.float32), dtype=np.float32)
            run_dwm(args.inference_py, args.checkpoint_path, str(tmp6), str(pred_score_png))

        # Load prob01 (NO threshold)
        pred_prob01 = load_pred_png_prob01(str(pred_score_png))
        np.save(str(pred_prob_npy), pred_prob01)
        save_prob_tiff(pred_prob01, pred_prob_tif)
        save_prob_bluewhite_png(pred_prob01, pred_prob_bluewhite)

        # Save NDWI mask png
        save_binary_png(ndwi_mask, ndwi_mask_png)

        # Choose threshold ONLY for metrics/plots
        if args.pred_thr is None:
            chosen_thr, best_m, curve_df = sweep_thresholds(pred_prob01, ndwi_mask, step=args.sweep_step)
            m = best_m
        else:
            chosen_thr = float(args.pred_thr)
            pred_bin = pred_prob01 >= chosen_thr
            m = metrics_binary(pred_bin, ndwi_mask)
            curve_df = pd.DataFrame([{"thr": chosen_thr, **m}])

        # Optional overlay on RGB
        rgb01 = quick_rgb(s2n, band_names)
        overlay_png = ""
        if rgb01 is not None:
            overlay_png_path = window_out / f"{window}_overlay_blue_on_rgb.png"
            save_overlay_on_rgb(rgb01, pred_prob01, overlay_png_path, max_alpha=0.65)
            overlay_png = str(overlay_png_path)

        # Plots
        make_plots(window_out, window, rgb01, ndwi, ndwi_mask, pred_prob01, chosen_thr, m, curve_df)

        # Save sweep csv if auto
        if args.pred_thr is None and len(curve_df) > 1:
            curve_df.to_csv(window_out / f"{window}_thr_sweep.csv", index=False)

        rows.append({
            "window": window,
            "sentinel_dir": str(sentinel_dir),
            "s2_path": str(s2_path),

            "pred_score_png": str(pred_score_png),
            "pred_prob_npy": str(pred_prob_npy),
            "pred_prob_tif": str(pred_prob_tif),
            "pred_prob_bluewhite_png": str(pred_prob_bluewhite),
            "overlay_png": overlay_png,

            "ndwi_mask_png": str(ndwi_mask_png),
            "chosen_pred_thr_for_metrics": chosen_thr,
            **m
        })

    df = pd.DataFrame(rows)
    csv_path = out_root / "summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # Global plots if multiple windows
    if len(df) >= 2:
        def parse_start(w):
            try:
                return datetime.strptime(w.split("_")[0], "%Y%m%d")
            except Exception:
                return None

        df["start_dt"] = df["window"].apply(parse_start)
        df2 = df.dropna(subset=["start_dt"]).sort_values("start_dt")
        if len(df2) >= 2:
            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(1,1,1)
            ax.plot(df2["start_dt"], df2["iou"], marker="o")
            ax.set_title("IoU (DeepWaterMap vs NDWI@0.0) over time")
            fig.tight_layout()
            fig.savefig(str(out_root / "iou_over_time.png"), dpi=150)
            plt.close(fig)

            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(1,1,1)
            ax.plot(df2["start_dt"], df2["pred_water_pct"]*100, marker="o", label="pred% (at chosen thr)")
            ax.plot(df2["start_dt"], df2["ndwi_water_pct"]*100, marker="o", label="ndwi%")
            ax.set_title("Water percentage over time")
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(out_root / "water_pct_over_time.png"), dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
