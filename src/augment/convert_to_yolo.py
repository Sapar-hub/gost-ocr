#!/usr/bin/env python3
"""Convert ground truth JSON to YOLO format."""

import json
from pathlib import Path
from PIL import Image

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "yolo_dataset"

CLASSES = ["stamp"]

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert [x, y, w, h] to YOLO format [x_center, y_center, w, h] normalized."""
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return x_center, y_center, w_norm, h_norm


def main():
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    json_files = list(GROUND_TRUTH_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        img_path_str = data["image_path"]
        if img_path_str.startswith("ground_truth/"):
            img_path_str = img_path_str.replace("ground_truth/", "")
        if img_path_str.startswith("fullpage/"):
            img_path_str = img_path_str.replace("fullpage/", "")

        img_path = GROUND_TRUTH_DIR / img_path_str
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path} (from: {data['image_path']})")
            continue

        with Image.open(img_path) as img:
            img_width, img_height = img.size

        bbox = data["stamp_bbox"]
        x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(bbox, img_width, img_height)

        output_name = img_path.stem

        label_file = output_labels / f"{output_name}.txt"
        with open(label_file, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        import shutil
        dest_img = output_images / f"{output_name}{img_path.suffix}"
        shutil.copy(img_path, dest_img)

        print(f"Converted: {output_name} (img: {img_width}x{img_height}, bbox: {bbox})")

    print(f"\nDone! Dataset saved to: {OUTPUT_DIR}")
    print(f"Images: {output_images}")
    print(f"Labels: {output_labels}")


if __name__ == "__main__":
    main()