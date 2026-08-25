import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
# Load YOLO Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚙️ Detection Settings")

mode = st.sidebar.radio(
    "Select Detection Mode",
    [
        "📷 Image",
        "🎥 Video",
        "📹 Live Webcam"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = YOLO("yolo26n.pt")
        self.confidence = 0.4

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            image,
            conf=self.confidence,
            verbose=False
        )

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# --------------------------------------------------
# Main Title
# --------------------------------------------------

st.title("Vehicle & Traffic Light Detection System")

st.write("Detect vehicles and traffic lights using a pretrained YOLO model.")


# ==================================================
# IMAGE DETECTION
# ==================================================

if mode == "📷 Image":

    st.header("📷 Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        if st.button("🔍 Detect Objects"):

            with st.spinner("Detecting objects..."):

                results = model.predict(
                    image,
                    conf=confidence
                )

            result = results[0]

            result_image = result.plot()

            with col2:

                st.subheader("Detection Result")

                st.image(
                    result_image,
                    channels="BGR",
                    use_container_width=True
                )

            # ------------------------------------------
            # Detection Information
            # ------------------------------------------

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence_score = float(
                        box.conf[0]
                    )

                    class_name = model.names[class_id]

                    detections.append(
                        {
                            "Object": class_name,
                            "Confidence": f"{confidence_score:.2%}"
                        }
                    )

            st.subheader("🔍 Detected Objects")

            if detections:

                st.dataframe(
                    detections,
                    use_container_width=True
                )

                st.success(
                    f"{len(detections)} object(s) detected."
                )

            else:

                st.warning(
                    "No objects detected."
                )


# ==================================================
# VIDEO DETECTION
# ==================================================

elif mode == "🎥 Video":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_video:

            temp_video.write(
                uploaded_video.read()
            )

            input_video_path = temp_video.name

        st.video(uploaded_video)

        if st.button("🔍 Detect Objects in Video"):

            output_placeholder = st.empty()

            progress_bar = st.progress(0)

            with st.spinner(
                "Processing video..."
            ):

                results = model.predict(
                    source=input_video_path,
                    conf=confidence,
                    save=True,
                    stream=True
                )

                total_frames = 0

                for result in results:

                    total_frames += 1

                    frame = result.plot()

                    output_placeholder.image(
                        frame,
                        channels="BGR",
                        use_container_width=True
                    )

                    # Progress indicator
                    if total_frames % 10 == 0:
                        progress_bar.progress(
                            min(
                                total_frames / 1000,
                                0.99
                            )
                        )

            progress_bar.progress(1.0)

            st.success(
                "Video processing completed!"
            )


# ==================================================
# Live Webcam
# ==================================================

elif mode == "📹 Live Webcam":

    st.header("📹 Real-Time Object Detection")

    st.write(
        "Allow camera access when your browser asks for permission."
    )

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


    # Clean temporary file
    try:
        os.remove(input_video_path)
    except OSError:
        pass