#!/usr/bin/env python
"""
Brain Tumor MRI — Keras inference (CPU-only)

- Loads `model.keras` (preferred) or falls back to a TensorFlow SavedModel folder.
- Uses `meta.json` for input size and class names.
- Predicts a single image (--image) or an entire folder (--folder).
- Folder mode writes a CSV of per-class probabilities.

Notes:
- The training graph already included EfficientNet preprocess_input.
  We therefore feed RGB images as float32 in the 0..255 range (no /255).
"""

# ---------- CPU-only + quiet logs (set BEFORE importing tensorflow) ----------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU even if a GPU exists
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence INFO logs
# If you want to silence oneDNN notices too, uncomment:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------- Imports ----------
import json, argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf


# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
KERAS_PATH      = HERE / "model.keras"
META_PATH       = HERE / "meta.json"
# Accept SavedModel either at project root or under brain_tumor_infer/
SAVED_DIR_ROOT  = HERE / "saved_model"
SAVED_DIR_SUB   = HERE / "brain_tumor_infer" / "saved_model"


# ---------- Metadata ----------
if not META_PATH.exists():
    raise FileNotFoundError(f"Missing meta.json next to infer.py: {META_PATH}")
with META_PATH.open("r") as f:
    META = json.load(f)

IMG_SIZE    = tuple(META["img_size"])          # e.g. (256, 256)
CLASS_NAMES = list(META["classes"])             # e.g. ["glioma","meningioma","notumor","pituitary"]


# ---------- Model loading (robust across Keras versions) ----------
def load_model_any():
    """
    Try to load native .keras first.
    If that fails (version/deserialization issues), load SavedModel via TFSMLayer.
    """
    print("TensorFlow:", tf.__version__)

    # 1) Try native Keras file
    if KERAS_PATH.exists():
        try:
            model = tf.keras.models.load_model(KERAS_PATH, compile=False)
            print(f"Loaded .keras: {KERAS_PATH.name}")
            return model
        except Exception as e:
            print("Failed to load .keras; will try SavedModel via TFSMLayer.\nReason:", e)

    # 2) Fallback to SavedModel via TFSMLayer (Keras 3 way)
    # Pick whichever SavedModel path exists
    if (SAVED_DIR_ROOT / "saved_model.pb").exists():
        sm_path = SAVED_DIR_ROOT
    elif (SAVED_DIR_SUB / "saved_model.pb").exists():
        sm_path = SAVED_DIR_SUB
    else:
        raise FileNotFoundError(
            "No model available. Place either:\n"
            f" - {KERAS_PATH.name} next to infer.py, or\n"
            f" - a SavedModel folder at '{SAVED_DIR_ROOT}/' or '{SAVED_DIR_SUB}/' containing saved_model.pb"
        )

    from keras.layers import TFSMLayer

    # Endpoint name can be "serve" (Keras export) or "serving_default"
    last_err = None
    for endpoint in ("serve", "serving_default"):
        try:
            tfsml = TFSMLayer(str(sm_path), call_endpoint=endpoint)
            break
        except Exception as e:
            last_err = e
            tfsml = None
    if tfsml is None:
        raise RuntimeError(f"Could not open SavedModel at {sm_path}\n{last_err}")

    # Wrap the SavedModel as a Keras model with a clean input signature
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3), dtype=tf.float32, name="input")
    out = tfsml(inp)
    if isinstance(out, dict):                   # handle dict outputs
        out = list(out.values())[0]
    model = tf.keras.Model(inp, out, name="brain_tumor_infer_tfsml")
    print(f"Loaded SavedModel via TFSMLayer: {sm_path}")
    return model


MODEL = load_model_any()


# ---------- Preprocessing & prediction ----------
def _prep_from_pil(pil_img: Image.Image) -> np.ndarray:
    """
    PIL image -> (1, H, W, 3) float32 in 0..255.
    Honors EXIF rotation to avoid sideways predictions.
    """
    im  = ImageOps.exif_transpose(pil_img).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(im, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def _predict_array(x: np.ndarray) -> tuple[int, np.ndarray]:
    """Run model on a (1, H, W, 3) array and return (top_idx, probs)."""
    probs = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, probs

def predict_one(path: str) -> dict:
    """Predict a single image file and return a result row (dict)."""
    with Image.open(path) as im:
        idx, probs = _predict_array(_prep_from_pil(im))
    return {
        "file": str(Path(path).resolve()),
        "pred": CLASS_NAMES[idx],
        "conf": float(probs[idx]),
        **{f"p_{c}": float(p) for c, p in zip(CLASS_NAMES, probs)},
    }


# ---------- CLI actions ----------
def run_image(path: str) -> None:
    r = predict_one(path)
    print(f"\nFile: {Path(path).name}")
    print(f"Top-1: {r['pred']} ({r['conf']:.2%})")
    for c in CLASS_NAMES:
        print(f"{c:>12}: {r[f'p_{c}']:.4f}")

def _iter_image_files(folder: str, recursive: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    pat = "**/*" if recursive else "*"
    for p in Path(folder).glob(pat):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)

def run_folder(folder: str, out_csv: str, batch: int = 16, recursive: bool = False) -> None:
    files = list(_iter_image_files(folder, recursive))
    if not files:
        print("No images found in:", folder)
        return

    rows: list[dict] = []
    i = 0
    while i < len(files):
        chunk = files[i:i+batch]

        # Load a batch, skipping unreadable files
        imgs, keep = [], []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB").resize(IMG_SIZE)
                    imgs.append(np.asarray(im, dtype=np.float32))
                keep.append(p)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping unreadable file: {p}")

        if imgs:
            x = np.stack(imgs, axis=0)           # (N, H, W, 3)
            probs = MODEL.predict(x, verbose=0)  # (N, C)
            for pth, prob in zip(keep, probs):
                k = int(np.argmax(prob))
                rows.append({
                    "file": str(Path(pth).resolve()),
                    "pred": CLASS_NAMES[k],
                    "conf": float(prob[k]),
                    **{f"p_{c}": float(v) for c, v in zip(CLASS_NAMES, prob)},
                })

        i += batch

    if not rows:
        print("No valid images processed.")
        return

    out_path = Path(out_csv)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} predictions → {out_path}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Brain Tumor MRI inference (Keras, CPU-only).")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--image",  help="Path to a single image")
    m.add_argument("--folder", help="Path to a folder of images")
    ap.add_argument("--out",      default="predictions.csv", help="CSV path for --folder")
    ap.add_argument("--batch",    type=int, default=16, help="Batch size for --folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders for --folder")
    args = ap.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_folder(args.folder, args.out, batch=args.batch, recursive=args.recursive)

if __name__ == "__main__":
    main()
