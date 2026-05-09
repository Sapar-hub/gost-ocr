#!/usr/bin/env python3
"""Generate synthetic training data in YOLO format."""

import random
import shutil
from pathlib import Path
from typing import Annotated

import typer
from fullpage_generator import generate_full_page_synthetic

app = typer.Typer()


def convert_to_yolo_format(bbox: list, img_w: int, img_h: int) -> tuple:
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


@app.command()
def main(
    output_dir: Annotated[Path, typer.Argument(help="Output directory for YOLO dataset")] = Path(
        "gost_ocr/datasets/train"
    ),
    paper_sizes: Annotated[
        list[str], typer.Option(help="Paper sizes to generate")
    ] = ["A0", "A1", "A2", "A3", "A4"],
    dpi_values: Annotated[
        list[int], typer.Option(help="DPI values to generate")
    ] = [200, 300],
    num_samples: Annotated[
        int, typer.Option(help="Samples per paper size/DPI combination")
    ] = 20,
    val_ratio: Annotated[
        float, typer.Option(help="Validation split ratio")
    ] = 0.15,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
):
    random.seed(seed)

    for d in [
        output_dir / "images" / "train",
        output_dir / "labels" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "val",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    for paper_size in paper_sizes:
        for dpi in dpi_values:
            temp_dir = output_dir / "temp" / f"{paper_size}_{dpi}dpi"
            temp_dir.mkdir(parents=True, exist_ok=True)

            metadata = generate_full_page_synthetic(
                output_dir=temp_dir,
                num_samples=num_samples,
                dpi_values=[dpi],
                paper_size=paper_size,
                orientation="portrait",
            )
            all_metadata.extend(metadata)

    random.shuffle(all_metadata)

    split_idx = int(len(all_metadata) * (1 - val_ratio))
    train_data = all_metadata[:split_idx]
    val_data = all_metadata[split_idx:]

    yolo_yaml = output_dir / "gost_stamp.yaml"
    yolo_yaml.write_text(
        """# YOLO dataset config
train: images/train
val: images/val

nc: 1
names: ['stamp']
"""
    )

    def process_split(data: list, split: str):
        images_subdir = output_dir / "images" / split
        labels_subdir = output_dir / "labels" / split

        for meta in data:
            img_path = Path(meta["image_path"])
            if not img_path.exists():
                continue

            img_name = f"{img_path.parent.name}_{img_path.stem}.png"
            label_name = img_name.replace(".png", ".txt")

            new_img_path = images_subdir / img_name
            new_label_path = labels_subdir / label_name

            shutil.copy2(img_path, new_img_path)

            from PIL import Image
            img = Image.open(img_path)
            img_w, img_h = img.size

            cx, cy, bw, bh = convert_to_yolo_format(meta["stamp_bbox"], img_w, img_h)
            with open(new_label_path, "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"  {split}: {len(data)} samples")

    print(f"\nSplitting data (val_ratio={val_ratio}):")
    process_split(train_data, "train")
    process_split(val_data, "val")

    shutil.rmtree(output_dir / "temp", ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Generated YOLO training data in: {output_dir}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"\nTo train YOLO:")
    print(f"  uv run yolo detect train data={yolo_yaml} model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    app()