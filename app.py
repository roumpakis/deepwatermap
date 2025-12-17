import io
import os
import sys
import uuid
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile as tiff
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, Response

APP_VERSION = "0.2.0"

# Defaults for Sentinel-2 band naming (if not provided)
S2_12_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
S2_13_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

# DeepWaterMap expects: Blue, Green, Red, NIR, SWIR1, SWIR2
DWM_REQUIRED = ["B2","B3","B4","B8","B11","B12"]

# Config via env vars
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/cp.135.ckpt")
INFERENCE_SCRIPT = os.environ.get("INFERENCE_SCRIPT", "inference.py")
PORT = int(os.environ.get("PORT", "8080"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "25000000"))

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RESULTS = STATIC / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC))


def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Accepts (H,W,C) or (C,H,W) and returns (H,W,C)."""
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
    # heuristic: if first dim looks like channels
    if arr.shape[0] in (2, 6, 10, 12, 13) and arr.shape[1] > 16 and arr.shape[2] > 16:
        return np.transpose(arr, (1, 2, 0))
    return arr


def normalize_s2(x: np.ndarray) -> np.ndarray:
    """Heuristic: divide by 10000 if reflectance-like scaling, clip to [0,1]."""
    x = x.astype(np.float32)
    mx = float(np.nanmax(x))
    if 1.5 < mx <= 20000:
        x = x / 10000.0
    return np.clip(x, 0.0, 1.0)


def parse_band_order(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    bands = [b.strip() for b in s.split(",") if b.strip()]
    return bands or None


def infer_default_band_order(c: int) -> Optional[List[str]]:
    if c == 12:
        return S2_12_BANDS
    if c == 13:
        return S2_13_BANDS
    if c == 6:
        return DWM_REQUIRED[:]  # already in DWM order
    return None


def select_bands(hwc: np.ndarray, band_names: List[str], required: List[str]) -> np.ndarray:
    if hwc.shape[2] != len(band_names):
        raise ValueError(f"band_order length {len(band_names)} != channels {hwc.shape[2]}")
    idx = {b: i for i, b in enumerate(band_names)}
    missing = [b for b in required if b not in idx]
    if missing:
        raise ValueError(f"Missing required bands: {missing}. Available={band_names}")
    return np.stack([hwc[:, :, idx[b]] for b in required], axis=2)


def compute_ndwi(hwc: np.ndarray, band_names: List[str], thr: float) -> Tuple[np.ndarray, np.ndarray]:
    idx = {b: i for i, b in enumerate(band_names)}
    if "B3" not in idx or "B8" not in idx:
        raise ValueError("NDWI requires B3 (Green) and B8 (NIR). Provide correct band_order.")
    g = hwc[:, :, idx["B3"]].astype(np.float32)
    n = hwc[:, :, idx["B8"]].astype(np.float32)
    ndwi = (g - n) / (g + n + 1e-6)
    mask = ndwi > float(thr)
    return ndwi, mask


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
    return np.clip(rgb, 0.0, 1.0)


def run_dwm(tmp6_tif: Path, out_png: Path) -> None:
    script = str(ROOT / INFERENCE_SCRIPT) if not os.path.isabs(INFERENCE_SCRIPT) else INFERENCE_SCRIPT
    cmd = [
        sys.executable, script,
        "--checkpoint_path", CHECKPOINT_PATH,
        "--image_path", str(tmp6_tif),
        "--save_path", str(out_png),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "DeepWaterMap inference failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def load_pred_prob01(pred_png: Path) -> np.ndarray:
    im = Image.open(str(pred_png)).convert("L")
    arr = np.array(im).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_binary_png(mask: np.ndarray, path: Path) -> None:
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(str(path))


def save_prob_bluewhite_png(prob01: np.ndarray, path: Path) -> None:
    """
    White→Blue colormap:
      p=0 -> white (255,255,255)
      p=1 -> blue  (0,0,255)
    """
    p = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    r = (1.0 - p)
    g = (1.0 - p)
    b = np.ones_like(p)
    rgb = np.stack([r, g, b], axis=2)
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(str(path))


def save_overlay_blue_on_rgb(rgb01: np.ndarray, prob01: np.ndarray, path: Path, max_alpha: float = 0.65) -> None:
    base = np.clip(rgb01.astype(np.float32), 0.0, 1.0)
    p = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    overlay = np.zeros_like(base)
    overlay[..., 2] = 1.0  # blue
    alpha = (max_alpha * p)[..., None]
    out = base * (1.0 - alpha) + overlay * alpha
    out = np.clip(out, 0.0, 1.0)
    Image.fromarray((out * 255).astype(np.uint8), mode="RGB").save(str(path))


def save_diff_mask_png(pred_bin: np.ndarray, ndwi_bin: np.ndarray, path: Path) -> None:
    """
    Diff colors:
      TN: white
      TP: blue
      FP: red
      FN: orange
    """
    pred = pred_bin.astype(bool)
    ref = ndwi_bin.astype(bool)
    out = np.ones((pred.shape[0], pred.shape[1], 3), dtype=np.uint8) * 255  # TN white

    tp = np.logical_and(pred, ref)
    fp = np.logical_and(pred, ~ref)
    fn = np.logical_and(~pred, ref)

    out[tp] = (0, 0, 255)
    out[fp] = (255, 0, 0)
    out[fn] = (255, 165, 0)

    Image.fromarray(out, mode="RGB").save(str(path))


def metrics_binary(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
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
        "iou": float(iou), "f1": float(f1),
        "precision": float(precision), "recall": float(recall),
        "accuracy": float(acc), "diff_pct": float(diff_pct),
        "pred_water_pct": float(pred.sum() / (total + 1e-9)),
        "ndwi_water_pct": float(ref.sum() / (total + 1e-9)),
        "fp_pct": float(fp / (total + 1e-9)),
        "fn_pct": float(fn / (total + 1e-9)),
    }


def sweep_thresholds(pred_prob01: np.ndarray, ref_mask: np.ndarray, step: float = 0.01):
    thrs = np.arange(0.0, 1.0 + 1e-9, step)
    best_thr = 0.5
    best_m = None
    for t in thrs:
        m = metrics_binary(pred_prob01 >= t, ref_mask)
        if best_m is None or m["f1"] > best_m["f1"]:
            best_m = m
            best_thr = float(t)
    return best_thr, best_m


@app.get("/")
def home():
    return send_from_directory(str(STATIC), "index.html")


@app.get("/health")
def health():
    return Response("ok", status=200, mimetype="text/plain")


@app.get("/results/<job_id>/<path:filename>")
def get_result(job_id: str, filename: str):
    d = RESULTS / job_id
    return send_from_directory(str(d), filename, as_attachment=False)


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "missing file field 'file' (multipart/form-data)"}), 400

    f = request.files["file"]
    filename = (f.filename or "input").lower()
    raw = f.read()

    ndwi_thr = float(request.form.get("ndwi_thr", "0.0"))
    pred_thr_raw = (request.form.get("pred_thr", "") or "").strip()
    pred_thr = None if pred_thr_raw == "" else float(pred_thr_raw)
    sweep_step = float(request.form.get("sweep_step", "0.01"))
    band_order = parse_band_order(request.form.get("band_order", ""))

    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded
    up_ext = ".npy" if filename.endswith(".npy") else (".tif" if filename.endswith(".tif") or filename.endswith(".tiff") else "")
    uploaded_path = out_dir / f"uploaded{up_ext}"
    with open(uploaded_path, "wb") as out:
        out.write(raw)

    try:
        # Load input
        if filename.endswith(".npy"):
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            arr = to_hwc(arr)
        elif filename.endswith(".tif") or filename.endswith(".tiff"):
            arr = tiff.imread(io.BytesIO(raw))
            arr = to_hwc(arr)
        else:
            return jsonify({"error": "Unsupported file type. Upload .npy or .tif/.tiff"}), 400

        h, w, c = arr.shape
        if h * w > MAX_PIXELS:
            return jsonify({"error": f"Image too large: {h}x{w}. Increase MAX_PIXELS env if needed."}), 413

        if band_order is None:
            band_order = infer_default_band_order(c)
        if band_order is None:
            return jsonify({"error": f"Cannot infer band order for C={c}. Provide band_order in UI."}), 400

        arrn = normalize_s2(arr)

        # NDWI + mask
        _, ndwi_mask = compute_ndwi(arrn, band_order, ndwi_thr)

        # 6-band for DWM
        s2_6 = select_bands(arrn, band_order, DWM_REQUIRED)

        # Run DWM
        pred_score_png = out_dir / "pred_score.png"
        with tempfile.TemporaryDirectory() as td:
            tmp6 = Path(td) / "tmp_s2_6band.tif"
            tiff.imwrite(str(tmp6), s2_6.astype(np.float32), dtype=np.float32)
            run_dwm(tmp6, pred_score_png)

        # Prob map [0,1]
        pred_prob01 = load_pred_prob01(pred_score_png)
        np.save(str(out_dir / "pred_prob01.npy"), pred_prob01)
        tiff.imwrite(str(out_dir / "pred_prob01.tif"), pred_prob01.astype(np.float32), dtype=np.float32)

        # Visuals
        save_prob_bluewhite_png(pred_prob01, out_dir / "pred_prob_bluewhite.png")
        save_binary_png(ndwi_mask, out_dir / "ndwi_mask.png")

        # threshold only for binary + diff + metrics (vs NDWI)
        if pred_thr is None:
            chosen_thr, m = sweep_thresholds(pred_prob01, ndwi_mask, step=sweep_step)
            thr_mode = "auto_best_f1_vs_ndwi"
        else:
            chosen_thr = float(pred_thr)
            m = metrics_binary(pred_prob01 >= chosen_thr, ndwi_mask)
            thr_mode = "fixed"

        pred_bin = pred_prob01 >= chosen_thr
        save_binary_png(pred_bin, out_dir / "pred_bin.png")
        save_diff_mask_png(pred_bin, ndwi_mask, out_dir / "diff_mask.png")

        rgb01 = quick_rgb(arrn, band_order)
        if rgb01 is not None:
            Image.fromarray((rgb01 * 255).astype(np.uint8), mode="RGB").save(str(out_dir / "rgb.png"))
            save_overlay_blue_on_rgb(rgb01, pred_prob01, out_dir / "overlay.png", max_alpha=0.65)
        else:
            shutil.copyfile(str(out_dir / "pred_prob_bluewhite.png"), str(out_dir / "overlay.png"))

        # zip outputs
        zip_path = out_dir / "outputs.zip"
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.iterdir():
                if p.is_file():
                    z.write(str(p), arcname=p.name)

        def url(name: str) -> str:
            return f"/results/{job_id}/{name}"

        return jsonify({
            "job_id": job_id,
            "inputs": {
                "filename": filename,
                "shape": [int(h), int(w), int(c)],
                "band_order": band_order,
                "ndwi_thr": ndwi_thr,
                "pred_thr_used": chosen_thr,
                "pred_thr_mode": thr_mode,
                "sweep_step": sweep_step if pred_thr is None else None,
            },
            "metrics_vs_ndwi": m,
            "outputs": {
                "rgb_png": url("rgb.png") if (out_dir / "rgb.png").exists() else None,
                "overlay_png": url("overlay.png"),
                "pred_prob_bluewhite_png": url("pred_prob_bluewhite.png"),
                "pred_prob01_npy": url("pred_prob01.npy"),
                "pred_prob01_tif": url("pred_prob01.tif"),
                "pred_bin_png": url("pred_bin.png"),
                "ndwi_mask_png": url("ndwi_mask.png"),
                "diff_mask_png": url("diff_mask.png"),
                "zip": url("outputs.zip"),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e), "job_id": job_id}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
