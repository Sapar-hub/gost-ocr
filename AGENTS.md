# Agents Instructions

## Key Commands

```bash
# Run full pipeline (preprocess -> localize -> extract)
uv run gost-ocr pipeline /path/to/images/

# YOLO detection only (standalone)
uv run gost-ocr detect /path/to/images/

# Localization only (OpenCV)
uv run gost-ocr localize /path/to/images/ --detector=opencv

# Evaluation against ground truth
uv run gost-ocr evaluate src/gost_ocr/datasets/test/ --output-dir output/
```

## Project Structure

```
src/gost_ocr/
├── cli.py                 # Typer CLI entrypoint
├── config.py            # Config (paths, constants)
├── preprocessing.py     # Image preprocessing
├── detection/           # Unified detection
│   ├── yolo.py          # YoloDetector
│   ├── opencv.py        # OpenCvDetector
│   └── factory.py       # get_detector()
├── extraction.py        # EasyOCR text extraction
├── evaluation.py       # Metrics (IoU, CER, WER)
├── models/yolo/best.pt  # Trained YOLO weights
├── datasets/
│   ├── train/          # Synthetic training data
│   └── test/           # Test images + GT labels
└── tests/
    ├── analyze_comparison.py  # YOLO vs OpenCV comparison
    └── generate_visualization.py
```

## Important Flags

- `--detector`: Detection method - `auto` (YOLO+fallback), `yolo`, `opencv`
- `--flip`, `-f`: Try all rotations (0°, 90°, 180°, 270°)
- `--debug`, `-d`: Save intermediate images
- `--roi`: Region of interest - `top`, `bottom`, `left`, `right`, `corners`, `auto`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip sync pyproject.toml
```

## YOLO Training

```bash
uv run yolo detect train \
    data=src/gost_ocr/datasets/train/gost_stamp.yaml \
    model=yolov8n.pt \
    epochs=100 \
    device=cpu
```

## YOLO Inference

```bash
uv run gost-ocr detect src/gost_ocr/datasets/test/images/
```

## Generate Visual Comparison

```bash
uv run python src/gost_ocr/tests/generate_visualization.py
```

## Test Data

- Training: `src/gost_ocr/datasets/train/`
- Test GT: `src/gost_ocr/datasets/test/labels/`
- YOLO predictions: `output/*_yolo.json`
- Visual comparison: `docs/article/comparison/`