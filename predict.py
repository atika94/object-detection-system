"""
predict.py — Run object detection inference on an image or video.

Usage:
    python predict.py --source <path_to_image_or_video>
    python predict.py --source dataset/images/test/sample.jpg
    python predict.py --source 0          # webcam

Notes:
    - Update MODEL_PATH to point to your trained weights when available.
    - Results (annotated images/video) are saved under runs/detect/predict/.
"""

import argparse
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"   # Replace with trained weights, e.g. models/best.pt
CONF_THRESH = 0.25          # Minimum confidence threshold for detections
IMGSZ      = 640            # Inference image size
DEVICE     = "cpu"          # Use "cuda" if a GPU is available
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Object Detection — Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, or webcam index (e.g. 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=CONF_THRESH,
        imgsz=IMGSZ,
        device=DEVICE,
        save=True,          # Save annotated output to disk
        show=False,         # Set True to open a live preview window
    )

    print(f"[INFO] Inference complete. Results saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
