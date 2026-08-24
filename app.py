"""
app.py — Streamlit web app for interactive object detection.

Usage:
    streamlit run app.py

Notes:
    - This app will be built out in a later phase.
    - Requires: pip install streamlit ultralytics opencv-python
"""

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection System",
    page_icon="🔍",
    layout="centered",
)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Object Detection System")
st.markdown(
    """
    Welcome to the **Object Detection System** powered by [Ultralytics YOLO](https://docs.ultralytics.com/).

    > **Status:** Project initialised — model training and full UI coming soon.

    ### What this app will do:
    - Upload an image and detect objects in real time.
    - Display bounding boxes, class labels, and confidence scores.
    - Show summary statistics for each detection run.
    """
)

st.info("🚧 Full functionality will be added after model training is complete.")
