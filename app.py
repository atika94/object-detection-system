"""
app.py — Streamlit web app for interactive object detection.

Usage:
    streamlit run app.py

Notes:
    - This app uses the fine-tuned model located at models/best.pt.
    - Requires: pip install streamlit ultralytics opencv-python Pillow
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
CONF_THRESH = 0.25
# ─────────────────────────────────────────────────────────────────────────────

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource
def load_model():
    """Load the YOLOv8 model once and cache it."""
    return YOLO(MODEL_PATH)

def main():
    st.title("🔍 Vehicle Detection System")
    st.markdown("Upload an image to detect vehicles (cars, motorcycles, buses, trucks) using the fine-tuned YOLOv8 model.")

    model = load_model()

    st.sidebar.title("Settings")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONF_THRESH, 0.05)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.markdown("---")
        
        # Inference
        with st.spinner("Detecting vehicles..."):
            results = model.predict(image_np, conf=conf_thresh, imgsz=640)
        
        # Parse results
        result = results[0]
        annotated_img = result.plot() # Returns BGR numpy array
        
        # Convert BGR to RGB for Streamlit
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Display images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_img_rgb, use_column_width=True)

        # Summary Statistics
        st.markdown("### 📊 Detection Summary")
        
        boxes = result.boxes
        if len(boxes) == 0:
            st.info("No vehicles detected.")
        else:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            
            # Count detected classes
            names = model.names
            class_names = [names[cls_id] for cls_id in class_ids]
            counts = Counter(class_names)
            
            # Display stats in columns
            stat_cols = st.columns(len(counts))
            for i, (cls_name, count) in enumerate(counts.items()):
                with stat_cols[i]:
                    st.metric(label=cls_name.capitalize(), value=count)
            
            # Detailed breakdown
            st.markdown("#### Details")
            details = []
            for i in range(len(boxes)):
                details.append(f"- **{class_names[i]}**: {confidences[i]:.0%} confidence")
            st.markdown("\n".join(details))

if __name__ == "__main__":
    main()
