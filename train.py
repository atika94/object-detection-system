"""
train.py — Fine-tune a pretrained YOLO model on a custom dataset.

Usage:
    python train.py

Notes:
    - Update MODEL, DATA, EPOCHS, and IMGSZ as needed.
    - Trained weights will be saved under runs/detect/train/weights/.
"""

from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
MODEL  = "yolov8n.pt"    # Pretrained model: n=nano, s=small, m=medium, l=large
DATA   = "data.yaml"     # Dataset config file
EPOCHS = 50              # Training epochs (agreed: 50)
IMGSZ  = 640             # Input image size (pixels)
BATCH  = 16              # Images per batch (reduce if you run out of RAM)
DEVICE = "cpu"           # Use "cuda" if a GPU is available
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"[INFO] Loading model: {MODEL}")
    model = YOLO(MODEL)

    print(f"[INFO] Starting training for {EPOCHS} epoch(s)...")
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
    )

    print("[INFO] Training complete.")
    print(f"[INFO] Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
