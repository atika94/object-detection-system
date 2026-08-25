"""
download_dataset.py — One-time script to prepare the vehicle dataset.

Source  : COCO128 (official Ultralytics dataset — no account needed, ~25 MB)
Classes : car, motorcycle, bus, truck  (remapped from COCO IDs)
Output  : dataset/images/{train,val,test}/  &  dataset/labels/{train,val,test}/
          data.yaml updated automatically.

Usage:
    python download_dataset.py
"""

import shutil
import yaml
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
# COCO 0-indexed class ID → our class ID
COCO_TO_OUR = {
    2: 0,   # car        → 0
    3: 1,   # motorcycle → 1
    5: 2,   # bus        → 2
    7: 3,   # truck      → 3
}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
SPLIT_RATIO = (0.8, 0.1, 0.1)   # train / val / test
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent


def download_coco128() -> Path:
    """Download COCO128 via Ultralytics and return its root directory."""
    from ultralytics.data.utils import check_det_dataset

    print("[INFO] Checking / downloading COCO128 (this may take a moment)...")
    data = check_det_dataset("coco128.yaml")
    coco_root = Path(data["path"])
    print(f"[INFO] COCO128 ready at: {coco_root}")
    return coco_root


def filter_vehicles(coco_root: Path) -> list:
    """
    Walk every label file in COCO128 train split.
    Keep only annotations whose class ID is in COCO_TO_OUR,
    remap to our IDs, and return (image_path, [label_lines]) pairs.
    """
    images_dir = coco_root / "images" / "train2017"
    labels_dir = coco_root / "labels" / "train2017"

    vehicle_data = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            lines = f.readlines()

        filtered = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            coco_id = int(parts[0])
            if coco_id in COCO_TO_OUR:
                our_id = COCO_TO_OUR[coco_id]
                # Keep bounding box coords as-is (already normalised)
                filtered.append(f"{our_id} {' '.join(parts[1:])}\n")

        if filtered:
            vehicle_data.append((img_path, filtered))

    print(f"[INFO] {len(vehicle_data)} images contain vehicle annotations.")
    return vehicle_data


def split_and_copy(vehicle_data: list) -> dict:
    """
    Split vehicle_data into train / val / test according to SPLIT_RATIO,
    then copy images and write remapped label files to dataset/.
    """
    n       = len(vehicle_data)
    n_train = int(n * SPLIT_RATIO[0])
    n_val   = int(n * SPLIT_RATIO[1])

    splits = {
        "train": vehicle_data[:n_train],
        "val":   vehicle_data[n_train : n_train + n_val],
        "test":  vehicle_data[n_train + n_val :],
    }

    counts = {}
    for split_name, items in splits.items():
        img_dst = PROJECT_ROOT / "dataset" / "images" / split_name
        lbl_dst = PROJECT_ROOT / "dataset" / "labels" / split_name

        # Clean out old placeholder .gitkeep and any stale files
        shutil.rmtree(img_dst, ignore_errors=True)
        shutil.rmtree(lbl_dst, ignore_errors=True)
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path, label_lines in items:
            # Copy the image
            shutil.copy(img_path, img_dst / img_path.name)
            # Write the remapped label file
            lbl_file = lbl_dst / (img_path.stem + ".txt")
            with open(lbl_file, "w") as f:
                f.writelines(label_lines)

        counts[split_name] = len(items)
        print(f"  [{split_name:5s}]  {len(items):3d} images")

    return counts


def update_data_yaml():
    """Rewrite data.yaml with the correct class names and paths."""
    data = {
        "train": "dataset/images/train",
        "val":   "dataset/images/val",
        "test":  "dataset/images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = PROJECT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[INFO] data.yaml → nc={data['nc']}, names={data['names']}")


def main():
    print("=" * 55)
    print("  Vehicle Dataset Preparation")
    print("  Source  : COCO128 (Ultralytics, no auth needed)")
    print("  Classes : car, motorcycle, bus, truck")
    print("  Split   : 80 / 10 / 10  (train / val / test)")
    print("=" * 55 + "\n")

    coco_root    = download_coco128()
    vehicle_data = filter_vehicles(coco_root)

    print("\n[INFO] Splitting and copying files to dataset/ ...")
    counts = split_and_copy(vehicle_data)
    update_data_yaml()

    total = sum(counts.values())
    print(f"\n✅  Dataset ready!  Total: {total} images")
    print("   Run training with:  python train.py\n")


if __name__ == "__main__":
    main()
