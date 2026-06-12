# Agents Instructions

## Key Commands

```bash
# Preprocess images only (deskew + flip + ROI)
uv run gost-ocr preprocess /path/to/images/

# Run full pipeline (preprocess -> localize -> extract)
uv run gost-ocr pipeline /path/to/images/

# YOLO detection only (standalone)
uv run gost-ocr detect /path/to/images/

# Localization only (OpenCV)
uv run gost-ocr localize /path/to/images/ --detector=opencv

# Evaluation against ground truth
uv run gost-ocr evaluate src/gost_ocr/datasets/yolo/images/test/ \
    --output-dir output/
```

## Project Structure

```
src/gost_ocr/
├── cli.py                 # Typer CLI entrypoint
├── config.py             # Config (paths, constants, DPI values)
├── preprocessing.py      # Image preprocessing (deskew, flip, ROI)
├── localization.py       # OpenCV contour-based localization
├── extraction.py        # EasyOCR text extraction
├── evaluation.py        # Metrics (IoU, CER, WER)
├── detection/           # Unified detection
│   ├── base.py         # Abstract Detector class, DetectionResult dataclass
│   ├── yolo.py         # YoloDetector
│   ├── opencv.py       # OpenCvDetector
│   └── factory.py      # get_detector(), get_detector_info()
├── datasets/yolo/
│   └── gost_stamp.yaml  # YOLO training config
└── models/yolo/
    └── best.pt          # Trained YOLO weights

src/gost_ocr/benchmark/ # Comparison module
├── constants.py         # ROI mapping, paths
├── metrics.py           # IoU, Precision/Recall/F1
├── compare.py           # Main comparison script
└── visualize.py         # Visualization generator
```

## Important Flags

**Global flags** (available for all commands):
- `-r`, `--recursive`: Process nested folders recursively
- `--flip`, `-f`: Try all rotations (0°, 90°, 180°, 270°)
- `--debug`, `-d`: Save intermediate images

**Detection flags:**
- `--detector`: Detection method - `auto` (YOLO+fallback), `yolo`, `opencv`
- `--filter-by-size/--no-filter-by-size`: Filter stamps by size config

**Preprocessing/Localization flags:**
- `--roi`: Region of interest - `top`, `bottom`, `left`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right`, `full_page`, `corners`, `auto`
- `--dpi`: DPI setting - `auto`, 200, 300, 400, 600

**Evaluate flags:**
- `--output-dir`, `-o`: Output directory for results
- `--save-report`, `-s`: Save evaluation report to JSON
- `--iou-threshold`: Minimum IoU for localization match (default: 0.5)
- `--cer-threshold`: Maximum CER for recognition (default: 0.1)
- `--wer-threshold`: Maximum WER for recognition (default: 0.2)

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
    data=src/gost_ocr/datasets/yolo/gost_stamp.yaml \
    model=yolov8n.pt \
    epochs=100 \
    device=cpu
```

## YOLO Inference

```bash
uv run gost-ocr detect src/gost_ocr/datasets/yolo/images/test/
```

## Benchmark (YOLO vs OpenCV comparison)

```bash
uv run python -m src.gost_ocr.benchmark.compare
```

## Test Data

- Test images: `src/gost_ocr/datasets/images/test/`
- Ground truth: `src/gost_ocr/datasets/labels/test/`
- Model: `src/gost_ocr/models/yolo/best.pt`
- YAML config: `src/gost_ocr/datasets/yolo/gost_stamp.yaml`
- YOLO predictions: `output/*_yolo.json`
- Visual comparison: `docs/article/comparison/`
