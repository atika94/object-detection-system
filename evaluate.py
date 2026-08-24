"""
evaluate.py — Evaluate a trained YOLO model on the test set.

Usage:
    python evaluate.py

Metrics reported:
    - mAP50    : Mean Average Precision at IoU threshold 0.50
    - mAP50-95 : Mean Average Precision averaged over IoU thresholds 0.50–0.95
    - Precision, Recall

Notes:
    - Update MODEL_PATH to point to your trained weights.
    - Results are saved under runs/detect/val/.
"""

from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"   # Path to trained weights
DATA       = "data.yaml"        # Dataset config file
IMGSZ      = 640                # Evaluation image size
DEVICE     = "cpu"              # Use "cuda" if a GPU is available
SPLIT      = "test"             # Dataset split to evaluate: "val" or "test"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[INFO] Evaluating on '{SPLIT}' split...")
    metrics = model.val(
        data=DATA,
        imgsz=IMGSZ,
        device=DEVICE,
        split=SPLIT,
    )

    print("\n── Evaluation Results ──────────────────────────────────")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
