# model_runtime.py
# CPU-only TensorFlow loader: prefer model.keras, fallback to SavedModel via TFSMLayer (Keras 3)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Optional: silence oneDNN notice
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

HERE = Path(__file__).resolve().parent
KERAS_PATH     = HERE / "model.keras"
META_PATH      = HERE / "meta.json"
SAVED_DIR_ROOT = HERE / "saved_model"
SAVED_DIR_SUB  = HERE / "brain_tumor_infer" / "saved_model"  # if you kept that layout

if not META_PATH.exists():
    raise FileNotFoundError(f"Missing meta.json next to app: {META_PATH}")
META = json.loads(META_PATH.read_text(encoding="utf-8"))
IMG_SIZE    = tuple(META["img_size"])
CLASS_NAMES = list(META["classes"])

def _load_model_any():
    # 1) Try native Keras file
    if KERAS_PATH.exists():
        try:
            m = tf.keras.models.load_model(KERAS_PATH, compile=False)
            print(f"[runtime] Loaded .keras: {KERAS_PATH.name}")
            return m
        except Exception as e:
            print("[runtime] .keras failed; will try SavedModel via TFSMLayer:", e)

    # 2) SavedModel via TFSMLayer (Keras 3 way)
    from keras.layers import TFSMLayer
    sm = None
    if (SAVED_DIR_ROOT / "saved_model.pb").exists():
        sm = SAVED_DIR_ROOT
    elif (SAVED_DIR_SUB / "saved_model.pb").exists():
        sm = SAVED_DIR_SUB
    else:
        raise FileNotFoundError(
            "No model found. Place model.keras or a SavedModel folder at ./saved_model/ (contains saved_model.pb)."
        )
    layer = None
    for endpoint in ("serve", "serving_default"):
        try:
            layer = TFSMLayer(str(sm), call_endpoint=endpoint)
            break
        except Exception:
            pass
    if layer is None:
        raise RuntimeError(f"Could not open SavedModel at {sm}")

    inp = tf.keras.Input(shape=(*IMG_SIZE, 3), dtype=tf.float32, name="input")
    out = layer(inp)
    if isinstance(out, dict):
        out = list(out.values())[0]
    m = tf.keras.Model(inp, out, name="brain_tumor_infer_tfsml")
    print(f"[runtime] Loaded SavedModel via TFSMLayer: {sm}")
    return m

MODEL = _load_model_any()

def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """EXIF-aware RGB resize -> float32 (1, H, W, 3); preprocess is embedded in the model."""
    im  = ImageOps.exif_transpose(pil_img).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(im, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict_pil(pil_img: Image.Image):
    """Return (top_label, top_conf, probs_dict) for a single image."""
    x = preprocess_pil(pil_img)
    probs = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
