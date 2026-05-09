#!/usr/bin/env python3
"""Compare YOLO vs OpenCV predictions against Ground Truth."""

import json
from pathlib import Path
from PIL import Image
import numpy as np

from ../../src/gost_ocr/config import YOLO_TEST_DIR, YOLO_TRAINED_MODEL, OUTPUT_DIR

YOLO_PRED_DIR = Path("runs/detect/predict-5/labels")

GT_DIR = YOLO_TEST_DIR / "labels"
TEST_IMAGES_DIR = YOLO_TEST_DIR / "images"
OPENCV_OUTPUT_DIR = OUTPUT_DIR


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1 + w1, x2 + w2) - xi
    hi = min(y1 + h1, y2 + h2) - yi

    if wi <= 0 or hi <= 0:
        return 0.0

    intersection = wi * hi
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0


def parse_gt_label(label_path, img_width, img_height):
    """Parse GT label (YOLO format normalized) to [x, y, w, h] in pixels."""
    with open(label_path) as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        _, x_center, y_center, w, h = map(float, parts[:5])

        x = (x_center - w / 2) * img_width
        y = (y_center - h / 2) * img_height
        w = w * img_width
        h = h * img_height

        bboxes.append([x, y, w, h])

    return bboxes


def find_matching_yolo_pred(gt_name, all_yolo_files):
    """Find matching YOLO prediction file based on image name in GT."""
    # GT files are like "0397768c-uzel-val-koleso.txt" - extract image name after hash
    gt_image_name = "-".join(gt_name.split("-")[1:])  # "uzel-val-koleso"
    
    for yolo_file in all_yolo_files:
        if gt_image_name in yolo_file.stem or yolo_file.stem in gt_image_name:
            return yolo_file
    
    # Fallback: check if any part matches
    gt_parts = gt_name.split("-")
    for yolo_file in all_yolo_files:
        for part in gt_parts:
            if len(part) > 5 and (part in yolo_file.stem or yolo_file.stem in part):
                return yolo_file
    return None


def get_yolo_predictions(yolo_file, orig_width, orig_height):
    """Parse YOLO predictions and scale to original image size."""
    if not yolo_file.exists():
        return []

    with open(yolo_file) as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        _, x_center, y_center, w, h = map(float, parts[:5])

        x = (x_center - w / 2) * orig_width
        y = (y_center - h / 2) * orig_height
        w = w * orig_width
        h = h * orig_height

        bboxes.append([x, y, w, h])

    return bboxes


def find_matching_opencv(gt_name, all_opencv_files):
    """Find matching OpenCV output file based on image name in GT."""
    # Extract image name from GT file
    gt_image_name = "-".join(gt_name.split("-")[1:])  # "uzel-val-koleso"
    
    for opencv_file in all_opencv_files:
        # OpenCV files have _output.json suffix, check stem without suffix
        opencv_stem = opencv_file.stem.replace("_output", "")
        if gt_image_name in opencv_stem or opencv_stem in gt_image_name:
            return opencv_file
    
    # Fallback
    gt_parts = gt_name.split("-")
    for opencv_file in all_opencv_files:
        opencv_stem = opencv_file.stem.replace("_output", "")
        for part in gt_parts:
            if len(part) > 5 and (part in opencv_stem or opencv_stem in part):
                return opencv_file
    return None


def main():
    results = []

    gt_files = sorted(GT_DIR.glob("*.txt"))
    all_yolo_files = list(YOLO_PRED_DIR.glob("*.txt"))
    all_opencv_files = list(OPENCV_OUTPUT_DIR.glob("*_output.json"))

    print(f"Found {len(gt_files)} GT labels, {len(all_yolo_files)} YOLO preds, {len(all_opencv_files)} OpenCV outputs")
    print("-" * 100)

    analyzed = set()

    for gt_file in gt_files:
        gt_name = gt_file.stem

        matching_yolo = find_matching_yolo_pred(gt_name, all_yolo_files)
        matching_opencv = find_matching_opencv(gt_name, all_opencv_files)

        image_candidates = []
        for f in TEST_IMAGES_DIR.iterdir():
            for part in gt_name.split("-"):
                if part in f.stem and len(part) > 5 and f.stem not in analyzed:
                    image_candidates.append(f)

        if not image_candidates:
            continue

        image_path = image_candidates[0]
        analyzed.add(image_path.stem)

        with Image.open(image_path) as img:
            orig_width, orig_height = img.size

        gt_bboxes = parse_gt_label(gt_file, orig_width, orig_height)
        if not gt_bboxes:
            continue

        yolo_bboxes = get_yolo_predictions(matching_yolo, orig_width, orig_height)

        opencv_bbox = None
        if matching_opencv:
            try:
                with open(matching_opencv) as f:
                    data = json.load(f)
                    if matching_opencv.name.endswith("_localization.json"):
                        if "stamp" in data and "bbox" in data["stamp"]:
                            opencv_bbox = data["stamp"]["bbox"]
                            print(f"    OpenCV (loc): {opencv_bbox} from {matching_opencv.name}")
                    elif "stamp_bbox" in data:
                        opencv_bbox = data["stamp_bbox"]
                        print(f"    OpenCV (out): {opencv_bbox} from {matching_opencv.name}")
            except Exception as e:
                print(f"    Error: {e}")

        best_yolo_iou = 0
        if yolo_bboxes:
            for yolo_bbox in yolo_bboxes:
                for gt_bbox in gt_bboxes:
                    iou = calculate_iou(yolo_bbox, gt_bbox)
                    if iou > best_yolo_iou:
                        best_yolo_iou = iou

        opencv_iou = calculate_iou(opencv_bbox, gt_bboxes[0]) if opencv_bbox else 0

        winner = "YOLO" if best_yolo_iou > opencv_iou else ("OpenCV" if opencv_iou > best_yolo_iou else "TIE")

        img_short = image_path.name[:35]
        print(f"{img_short:35} | YOLO IoU: {best_yolo_iou:.3f} ({len(yolo_bboxes)} preds) | OpenCV IoU: {opencv_iou:.3f} | {winner}")

        results.append({
            "image": image_path.name,
            "yolo_iou": best_yolo_iou,
            "yolo_preds": len(yolo_bboxes),
            "opencv_iou": opencv_iou,
            "winner": winner,
        })

    print("-" * 100)
    print("\n📊 SUMMARY:")
    print("=" * 80)

    yolo_ious = [r["yolo_iou"] for r in results]
    opencv_ious = [r["opencv_iou"] for r in results]

    yolo_mean = np.mean(yolo_ious)
    yolo_median = np.median(yolo_ious)
    opencv_mean = np.mean(opencv_ious)
    opencv_median = np.median(opencv_ious)

    print(f"Images analyzed: {len(results)}")
    print(f"\nYOLO:  Mean IoU = {yolo_mean:.3f}, Median = {yolo_median:.3f}")
    print(f"OpenCV: Mean IoU = {opencv_mean:.3f}, Median = {opencv_median:.3f}")

    yolo_wins = sum(1 for r in results if r["yolo_iou"] > r["opencv_iou"])
    opencv_wins = sum(1 for r in results if r["opencv_iou"] > r["yolo_iou"])
    ties = sum(1 for r in results if r["yolo_iou"] == r["opencv_iou"])

    print(f"\nYOLO wins: {yolo_wins}, OpenCV wins: {opencv_wins}, Ties: {ties}")

    with open(Path(__file__).parent.parent.parent / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to comparison_results.json")


if __name__ == "__main__":
    main()
