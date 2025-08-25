# app_streamlit.py  (CPU-only Streamlit UI)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional

# ---------- Imports ----------
# the below libraries 
import streamlit as st   # streamlit is used for web app  
import pandas as pd
import numpy as np  
from io import BytesIO  # this library is for byte data manipulation
from PIL import Image, ImageOps, UnidentifiedImageError  # this library is for image processing

# Reuse the robust runtime (already CPU-only and version-safe)
import model_runtime as rt

st.set_page_config(page_title="Brain Tumor MRI (CPU)", layout="wide")
st.title("Brain Tumor MRI Classifier (CPU)")
st.caption("Classes: " + ", ".join(rt.CLASS_NAMES))

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    batch_size = st.slider("Batch size (for multi files)", min_value=1, max_value=64, value=16, step=1)
    show_per_image_probs = st.checkbox("Show per-image probability bars", value=True)
    show_table = st.checkbox("Show results table", value=True)
    st.markdown("---")
    st.caption("Model artifacts searched at: `model.keras` or `./saved_model/`")

# Preprocess uploaded image bytes
def prep_bytes(data: bytes) -> np.ndarray:
    """Bytes -> (1, H, W, 3) float32, using the same preprocessing as runtime."""
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB").resize(rt.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0), img

# Predict a batch of images, return list of dicts with results
def predict_many(files, batch: int = 16):
    """Efficient batch inference with a fallback for unreadable files."""
    arrays, keep_imgs, names = [], [], []
    for f in files:
        try:
            arr, pil = prep_bytes(f.read())
            arrays.append(arr[0])   # remove batch dim for stacking
            keep_imgs.append(pil)
            names.append(f.name)
        except (UnidentifiedImageError, OSError) as e:
            st.warning(f"Skipping unreadable file: {f.name} ({e})")
    if not arrays:
        return [], []

    rows = []
    for i in range(0, len(arrays), batch):
        chunk = np.stack(arrays[i:i+batch], axis=0)  # (N, H, W, 3)
        probs = rt.MODEL.predict(chunk, verbose=0)
        for name, pil_img, prob in zip(names[i:i+batch], keep_imgs[i:i+batch], probs):
            k = int(np.argmax(prob))
            rows.append({
                "file": name,
                "pred": rt.CLASS_NAMES[k],
                "conf": float(prob[k]),
                **{f"p_{c}": float(v) for c, v in zip(rt.CLASS_NAMES, prob)},
                "_pil": pil_img,  # for display only
            })
    return rows, keep_imgs

# File uploader for input images
files = st.file_uploader("Upload JPG/PNG/BMP/TIF (you can select multiple)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)

if files:
    rows, _ = predict_many(files, batch=batch_size)

    if not rows:
        st.stop()

    # Display per-image cards (optional)
    if show_per_image_probs:
        st.subheader("Predictions")
        for r in rows:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2], vertical_alignment="center")
                with c1:
                    st.image(r["_pil"], caption=r["file"], use_container_width=True)
                with c2:
                    st.markdown(f"**{r['pred']}** · {r['conf']:.2%}")
                    # nice horizontal bars for per-class probs
                    for c in rt.CLASS_NAMES:
                        pct = r[f"p_{c}"]
                        st.progress(min(1.0, pct), text=f"{c}: {pct:.2%}")

    # Tabular results + CSV download
    if show_table:
        st.subheader("Results table")
        df = pd.DataFrame([{k:v for k,v in r.items() if k != "_pil"} for r in rows])
        # sort by confidence desc
        df = df.sort_values("conf", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Upload one or more images to get predictions.")
