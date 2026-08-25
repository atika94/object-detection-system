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
    st.markdown("Use the sidebar to upload an image or start the live real-time webcam feed to detect vehicles (cars, motorcycles, buses, trucks).")

    model = load_model()

    st.sidebar.title("Settings")
    input_source = st.sidebar.radio("Select Input Source", ["Image Upload", "Live Webcam"])
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONF_THRESH, 0.05)

    if input_source == "Image Upload":
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
                
                names = model.names
                class_names = [names[cls_id] for cls_id in class_ids]
                counts = Counter(class_names)
                
                stat_cols = st.columns(len(counts))
                for i, (cls_name, count) in enumerate(counts.items()):
                    with stat_cols[i]:
                        st.metric(label=cls_name.capitalize(), value=count)

    elif input_source == "Live Webcam":
        st.markdown("### 🎥 Live Real-Time Detection")
        st.info("Click **Start** to open your webcam. The feed will be processed frame-by-frame. Click **Stop** to end the stream.")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            run = st.checkbox("Start Webcam")
        
        FRAME_WINDOW = st.image([])
        
        if run:
            # 0 is usually the default built-in webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not access the webcam. Please ensure it's connected and not used by another application.")
                return

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to grab frame from webcam.")
                    break
                
                # Run YOLOv8 inference on the frame
                results = model.predict(frame, conf=conf_thresh, imgsz=640, verbose=False)
                result = results[0]
                
                # Plot bounding boxes
                annotated_frame = result.plot()
                
                # Convert BGR (OpenCV format) to RGB (Streamlit format)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update the image placeholder with the new frame
                FRAME_WINDOW.image(annotated_frame_rgb, channels="RGB")
                
            cap.release()
        else:
            st.write("Webcam is stopped.")

if __name__ == "__main__":
    main()
