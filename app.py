import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle & Traffic Light Detection",
    page_icon="🚦",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)


# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("🚦 Vehicle & Traffic Light Detection System")

st.write(
    "Detect vehicles and traffic lights using a pretrained YOLO model."
)


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):

        results = model.predict(
            image,
            conf=confidence
        )

    result = results[0]

    # Draw detections
    result_image = result.plot()

    with col2:
        st.subheader("Detection Result")
        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )

    # --------------------------------------------------
    # Detection Information
    # --------------------------------------------------

    st.subheader("🔍 Detected Objects")

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append(
                {
                    "Object": class_name,
                    "Confidence": f"{confidence_score:.2%}"
                }
            )

    if detections:

        st.dataframe(
            detections,
            use_container_width=True
        )

        st.success(
            f"{len(detections)} object(s) detected."
        )

    else:

        st.warning("No objects detected.")


else:

    st.info(
        "👆 Upload an image to start object detection."
    )