import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="Object Detection System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle & Traffic Light Detection")
st.write(
    "Detect vehicles and traffic lights using a pretrained YOLO model."
)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):

        results = model.predict(
            image,
            conf=confidence
        )

        result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(
            result_image,
            channels="BGR",
            use_container_width=True
        )

        st.success("Detection completed!")