#!/usr/bin/env python3
"""Convert ground truth JSON annotations to YOLO format."""

import json
import shutil
from pathlib import Path
from PIL import Image

GROUND_TRUTH_DIR = Path("ground_truth/yolo_training")
TRAIN_IMAGES_DIR = Path("src/gost_ocr/datasets/train/images/train")
TRAIN_LABELS_DIR = Path("src/gost_ocr/datasets/train/labels/train")

TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

json_files = list(GROUND_TRUTH_DIR.rglob("*.json"))
print(f"Found {len(json_files)} JSON files")

converted = 0
errors = 0

for json_file in json_files:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = Path(data["image_path"])
        if not image_path.is_absolute():
            image_path = Path("/home/saparch/playground/metrogiprotrans") / image_path

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            errors += 1
            continue

        bbox = data["stamp_bbox"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        img = Image.open(image_path)
        w, h = img.size

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        new_filename = image_path.stem + ".png"
        new_image_path = TRAIN_IMAGES_DIR / new_filename
        new_label_path = TRAIN_LABELS_DIR / (image_path.stem + ".txt")

        shutil.copy2(image_path, new_image_path)

        with open(new_label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1
        print(f"Converted: {new_filename}")

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        errors += 1

print(f"\nDone: {converted} converted, {errors} errors")