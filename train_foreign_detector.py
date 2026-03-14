"""
train_foreign_detector.py
Train a YOLOv8 model to detect foreign/unwanted objects (bottles, etc.)
using the images/ and labels/ dataset.

Steps:
  1. Remap sparse COCO class IDs → contiguous 0-indexed IDs
  2. Split into train/val (80/20)
  3. Generate data.yaml
  4. Fine-tune YOLOv8n on the dataset
"""

import os, shutil, random, yaml
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent
IMG_SRC   = ROOT / "images"
LBL_SRC   = ROOT / "labels"
DATASET   = ROOT / "dataset_foreign"          # output dir

TRAIN_IMG = DATASET / "images" / "train"
VAL_IMG   = DATASET / "images" / "val"
TRAIN_LBL = DATASET / "labels" / "train"
VAL_LBL   = DATASET / "labels" / "val"

# ── COCO-80 class names for IDs found in the labels ─────────────────────
# Original sparse IDs → human-readable names
# These are standard COCO IDs
COCO_NAMES = {
    0: "bottle",
    2: "cup",
    3: "fork",
    5: "knife",
    7: "bowl",
    8: "banana",
    9: "apple",
    10: "sandwich",
    11: "orange",
    12: "broccoli",
    13: "carrot",
    14: "hot_dog",
    15: "pizza",
    16: "donut",
    17: "cake",
    18: "chair",
    19: "couch",
    20: "potted_plant",
    22: "teddy_bear",
}

SPLIT_RATIO = 0.8
SEED = 42


def remap_and_split():
    """Remap labels → contiguous class IDs and create train/val split."""
    # Build remap: original → new contiguous ID
    original_ids = sorted(COCO_NAMES.keys())
    remap = {old: new for new, old in enumerate(original_ids)}
    names = [COCO_NAMES[old] for old in original_ids]
    print(f"[INFO] Class remap ({len(remap)} classes):")
    for old, new in remap.items():
        print(f"  COCO-{old} ({COCO_NAMES[old]}) → {new}")

    # Gather matched image-label pairs
    img_files = sorted(IMG_SRC.glob("*.jpg"))
    pairs = []
    for img_path in img_files:
        lbl_path = LBL_SRC / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    print(f"[INFO] Matched pairs: {len(pairs)}")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * SPLIT_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]
    print(f"[INFO] Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # Create dirs
    for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(img_src, lbl_src, img_dst_dir, lbl_dst_dir):
        shutil.copy2(img_src, img_dst_dir / img_src.name)
        # Remap label
        out_lines = []
        for line in lbl_src.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id in remap:
                parts[0] = str(remap[old_id])
                out_lines.append(" ".join(parts))
        (lbl_dst_dir / (img_src.stem + ".txt")).write_text("\n".join(out_lines) + "\n")

    print("[INFO] Copying train set...")
    for img, lbl in train_pairs:
        copy_pair(img, lbl, TRAIN_IMG, TRAIN_LBL)

    print("[INFO] Copying val set...")
    for img, lbl in val_pairs:
        copy_pair(img, lbl, VAL_IMG, VAL_LBL)

    # Write data.yaml
    data_yaml = {
        "path": str(DATASET),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path = DATASET / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] data.yaml written to {yaml_path}")
    return yaml_path, names


def train_model(yaml_path):
    """Fine-tune YOLOv8n on the foreign-object dataset."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # pretrained YOLOv8-nano
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        name="foreign_detector",
        project=str(ROOT / "runs"),
        patience=10,
        verbose=True,
    )
    # Copy best weights to project root
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "foreign_best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, dest)
        print(f"\n[SUCCESS] Best weights saved → {dest}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  Foreign Object Detector — YOLOv8 Training Pipeline")
    print("=" * 60)

    # Step 1-3: remap, split, create yaml
    yaml_path, class_names = remap_and_split()

    # Step 4: train
    print("\n[INFO] Starting YOLOv8 training...\n")
    results = train_model(yaml_path)
    print("\n[DONE] Training complete!")
