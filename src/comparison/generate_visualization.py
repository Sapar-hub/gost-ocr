#!/usr/bin/env python3
"""Generate visual comparison for YOLO vs OpenCV detection results."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from gost_ocr.config import YOLO_TEST_DIR, OUTPUT_DIR

VIS_OUTPUT_DIR = Path("docs/article/comparison")
VIS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COLORS = {
    "GT": (255, 0, 255),
    "YOLO": (0, 255, 0),
    "OpenCV": (0, 0, 255),
}


def calc_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi
    if wi <= 0 or hi <= 0:
        return 0.0
    return (wi*hi) / (w1*h1 + w2*h2 - wi*hi)


def parse_gt_label(label_path, img_w, img_h):
    with open(label_path) as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        _, xc, yc, ww, hh = map(float, parts[:5])
        x = (xc - ww/2) * img_w
        y = (yc - hh/2) * img_h
        w = ww * img_w
        h = hh * img_h
        bboxes.append([int(x), int(y), int(w), int(h)])
    return bboxes


def draw_rect(draw, x, y, w, h, color, label="", font=None):
    draw.rectangle([x-3, y-3, x+w+3, y+h+3], outline=(255,255,255), width=8)
    draw.rectangle([x, y, x+w, y+h], outline=color, width=6)
    if label and font:
        bbox = draw.textbbox((x+10, y+10), label, font=font)
        draw.rectangle([bbox[0]-10, bbox[1]-5, bbox[2]+10, bbox[3]+5], fill=(255,255,255))
        draw.text((x+10, y+10), label, fill=color, font=font)


def get_font(size):
    fonts = [
        "/usr/share/fonts/TTF/Rubik-VariableFont_wght.ttf",
        "/usr/share/fonts/TTF/JetBrainsMono-ExtraBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for f in fonts:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default()


def main():
    gt_dir = YOLO_TEST_DIR / "labels"
    img_dir = YOLO_TEST_DIR / "images"
    yolo_dir = OUTPUT_DIR
    opencv_dir = OUTPUT_DIR

    gt_files = sorted(gt_dir.glob("*.txt"))
    results = []

    for gt_file in gt_files:
        gt_stem = gt_file.stem
        parts = gt_stem.split("-", 1)
        gt_name = parts[1] if len(parts) > 1 else parts[0]

        img_path = None
        for f in img_dir.iterdir():
            if gt_name in f.stem:
                img_path = f
                break
        if not img_path:
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        gt_bboxes = parse_gt_label(gt_file, w, h)
        if not gt_bboxes:
            continue
        gt_bbox = gt_bboxes[0]

        yolo_file = None
        for f in yolo_dir.glob("*_yolo.json"):
            if gt_name in f.stem.replace("_yolo", ""):
                yolo_file = f
                break

        opencv_file = None
        for f in opencv_dir.glob("*_output.json"):
            if gt_name in f.stem.replace("_output", ""):
                opencv_file = f
                break

        yolo_iou = 0
        yolo_bbox = None
        if yolo_file:
            with open(yolo_file) as f:
                yolo_bbox = json.load(f)["bbox"]
                yolo_iou = calc_iou(yolo_bbox, gt_bbox)

        opencv_iou = 0
        opencv_bbox = None
        if opencv_file:
            with open(opencv_file) as f:
                oc = json.load(f)
            oc_bbox = oc.get("stamp_bbox", [0,0,0,0])
            if oc_bbox != [0,0,0,0]:
                opencv_bbox = oc_bbox
                opencv_iou = calc_iou(oc_bbox, gt_bbox)

        winner = "YOLO" if yolo_iou > opencv_iou else ("OPCV" if opencv_iou > yolo_iou else "TIE")
        results.append({
            "img_name": img_path.name,
            "gt_bbox": gt_bbox,
            "yolo_bbox": yolo_bbox,
            "opencv_bbox": opencv_bbox,
            "yolo_iou": yolo_iou,
            "opencv_iou": opencv_iou,
            "winner": winner,
        })

    header_font = get_font(60)
    label_font = get_font(40)

    for i, r in enumerate(results):
        img_path = img_dir / r["img_name"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        canvas_w = w * 2
        canvas_h = h * 2
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        gt_panel = img.copy()
        gt_draw = ImageDraw.Draw(gt_panel)
        draw_rect(gt_draw, *r["gt_bbox"], COLORS["GT"], "GT", label_font)

        yolo_panel = img.copy()
        yolo_draw = ImageDraw.Draw(yolo_panel)
        if r["yolo_bbox"]:
            draw_rect(yolo_draw, *r["yolo_bbox"], COLORS["YOLO"], "YOLO", label_font)
        draw_rect(yolo_draw, *r["gt_bbox"], COLORS["GT"], "", label_font)

        opencv_panel = img.copy()
        opencv_draw = ImageDraw.Draw(opencv_panel)
        if r["opencv_bbox"]:
            draw_rect(opencv_draw, *r["opencv_bbox"], COLORS["OpenCV"], "OpenCV", label_font)
        draw_rect(opencv_draw, *r["gt_bbox"], COLORS["GT"], "", label_font)

        all_panel = img.copy()
        all_draw = ImageDraw.Draw(all_panel)
        draw_rect(all_draw, *r["gt_bbox"], COLORS["GT"], "GT", label_font)
        if r["yolo_bbox"]:
            draw_rect(all_draw, *r["yolo_bbox"], COLORS["YOLO"], "YOLO", label_font)
        if r["opencv_bbox"]:
            draw_rect(all_draw, *r["opencv_bbox"], COLORS["OpenCV"], "OpenCV", label_font)

        canvas.paste(gt_panel, (0, 0))
        canvas.paste(yolo_panel, (w, 0))
        canvas.paste(opencv_panel, (0, h))
        canvas.paste(all_panel, (w, h))

        draw.text((10, 10), f"Ground Truth", fill=COLORS["GT"], font=header_font)
        draw.text((w+10, 10), f"YOLO IoU: {r['yolo_iou']:.2f}", fill=COLORS["YOLO"], font=header_font)
        draw.text((10, h+10), f"OpenCV IoU: {r['opencv_iou']:.2f}", fill=COLORS["OpenCV"], font=header_font)
        draw.text((w+10, h+10), f"Winner: {r['winner']}", fill=(255, 0, 0), font=header_font)

        short_name = f"img_{i+1:02d}_yolo{int(r['yolo_iou']*100)}_ocv{int(r['opencv_iou']*100)}.jpg"
        canvas.save(VIS_OUTPUT_DIR / short_name, quality=95)
        print(f"  {short_name}")

    print(f"\nSaved {len(results)} images to {VIS_OUTPUT_DIR}")

    with open(VIS_OUTPUT_DIR / "SUMMARY.md", "w") as f:
        f.write("# Comparison Results\n\n")
        f.write("| # | Image | YOLO IoU | OpenCV IoU | Winner |\n")
        f.write("|---|-------|----------|------------|--------|\n")
        for i, r in enumerate(results, 1):
            f.write(f"| {i} | {r['img_name']} | {r['yolo_iou']:.3f} | {r['opencv_iou']:.3f} | {r['winner']} |\n")
        f.write("\n## Summary\n\n")
        yolo_mean = sum(r["yolo_iou"] for r in results) / len(results)
        opencv_mean = sum(r["opencv_iou"] for r in results) / len(results)
        yolo_wins = sum(1 for r in results if r["winner"] == "YOLO")
        opencv_wins = sum(1 for r in results if r["winner"] in ("OPCV", "TIE"))
        f.write(f"- YOLO mean IoU: {yolo_mean:.3f} ({yolo_wins} wins)\n")
        f.write(f"- OpenCV mean IoU: {opencv_mean:.3f} ({opencv_wins} wins)\n")


if __name__ == "__main__":
    main()