# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
# --- Configuration ---
MODEL_PATH = "gear_model.pkl"
IMG_SIZE = 100
CATEGORIES = ["Bad Gear", "Good Gear"]

# --- Page Setup ---
st.set_page_config(page_title="Batch Gear Inspector", layout="centered")

st.title("Batch Gear Quality Inspector")
st.write("Select multiple gear images from your folder to analyze an entire batch at once!")

# --- Load the Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Error: Could not find gear_model.pkl. Please run train.py first.")
    st.stop()

# --- MULTIPLE Image Upload ---
uploaded_files = st.file_uploader(
    "Choose gear images...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)
# --- Process the Batch ---
if uploaded_files:
    st.write(f"### Analyzing {len(uploaded_files)} images...")
    st.divider()

    for file in uploaded_files:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(file)
            st.image(image, caption=file.name, width=150)
            
        with col2:
            file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img_array is None:
                st.error("Error processing this image.")
            else:
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                flattened_array = resized_array.flatten().reshape(1, -1)

                PASS_THRESHOLD = 0.90

                pred_idx = int(model.predict(flattened_array)[0])
                proba = model.predict_proba(flattened_array)[0]

                result_class = CATEGORIES[pred_idx]
                pred_conf = float(proba[pred_idx])

                if result_class == "Good Gear" and pred_conf >= PASS_THRESHOLD:
                    st.success(f"Result: {result_class} - PASS")
                elif result_class == "Bad Gear":
                    st.error(f"Result: {result_class} - FAIL")
                else:
                    st.warning(f"Result: {result_class} - REVIEW (low confidence)")

                st.info(f"AI Confidence: {pred_conf*100:.2f}%")
