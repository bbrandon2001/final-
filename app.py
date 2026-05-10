# -*- coding: utf-8 -*-
"""Streamlit app for Gear Quality Inspection (Bad vs Good).

How to run:
    streamlit run app.py

Expected files:
    gear_model.pkl   (trained scikit-learn model saved with joblib)

Expected folders (optional):
    dataset/bad , dataset/good (only needed for training, not for this app)

Notes:
- This app is PURE Python source code (no null bytes). If you previously got
  "source code cannot contain null bytes", you were likely trying to run a
  binary pickle that was accidentally named .py.
"""

import os
import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image

# ---------------- Configuration ----------------
MODEL_PATH = "gear_model.pkl"
IMG_SIZE = 100
CATEGORIES = ["Bad Gear", "Good Gear"]

# Safety: only PASS when Good probability is high
DEFAULT_PASS_THRESHOLD = 0.90


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def preprocess(img_gray: np.ndarray, img_size: int = IMG_SIZE, use_clahe: bool = True) -> np.ndarray:
    """Preprocess image to match training expectations.

    Always: grayscale -> resize.
    Optional: CLAHE + mild blur to reduce lighting noise.

    If your model was trained WITHOUT CLAHE, you can disable it in the UI.
    """
    if img_gray is None:
        raise ValueError("Image is None")

    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def to_sample(img_gray: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    img = preprocess(img_gray, IMG_SIZE, use_clahe=use_clahe)
    return img.flatten().reshape(1, -1)


def predict(model, sample: np.ndarray):
    """Return (pred_idx, proba_vector) using scikit-learn API."""
    pred_idx = int(model.predict(sample)[0])
    proba = model.predict_proba(sample)[0]
    return pred_idx, proba


# ---------------- UI ----------------
st.set_page_config(page_title="Gear Quality Inspector", layout="centered")
st.title("Gear Quality Inspector")
st.write("Upload one or more gear images. The app will classify each as **Good** or **Bad** with a safety PASS threshold.")

# Model loading
if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH}. Train your model first and place it next to app.py")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load {MODEL_PATH}: {e}")
    st.stop()

# Controls
with st.expander("Settings", expanded=True):
    pass_threshold = st.slider(
        "PASS threshold for 'Good' probability",
        min_value=0.50,
        max_value=0.99,
        value=float(DEFAULT_PASS_THRESHOLD),
        step=0.01,
        help="Only PASS when the model's probability of Good is at least this value. Higher = fewer defective parts pass."
    )
    use_clahe = st.checkbox(
        "Use CLAHE + mild denoise preprocessing",
        value=True,
        help="Helps with lighting changes. Disable only if your model was trained without it."
    )

uploaded_files = st.file_uploader(
    "Choose gear images...",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"### Analyzing {len(uploaded_files)} image(s)...")
    st.divider()

    for file in uploaded_files:
        col1, col2 = st.columns([1, 2])

        with col1:
            pil_img = Image.open(file)
            st.image(pil_img, caption=file.name, use_container_width=True)

        with col2:
            # Decode uploaded bytes with OpenCV
            file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
            img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img_gray is None:
                st.error("Error: Could not decode this image.")
                st.divider()
                continue

            try:
                sample = to_sample(img_gray, use_clahe=use_clahe)
                pred_idx, proba = predict(model, sample)
            except Exception as e:
                st.error(f"Error during preprocessing/prediction: {e}")
                st.divider()
                continue

            good_idx = CATEGORIES.index("Good Gear")
            good_prob = float(proba[good_idx])

            # Safety decision
            if good_prob >= pass_threshold:
                decision = "PASS"
                st.success(f"Decision: {decision}")
            else:
                decision = "FAIL / REVIEW"
                st.error(f"Decision: {decision}")

            pred_label = CATEGORIES[pred_idx]
            st.write(f"Prediction: **{pred_label}**")
            st.write(f"Good probability: **{good_prob*100:.2f}%**")
            st.caption(f"Probabilities → Bad: {proba[0]*100:.2f}% | Good: {proba[1]*100:.2f}%")

        st.divider()

else:
    st.info("Upload images to begin.")
