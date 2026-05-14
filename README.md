# GOST-OCR

**Automatic stamp localization and metadata extraction from GOST technical drawings.**

Scanned archival drawings contain all project metadata — document number, author, revision, date — locked inside a title block. This tool finds that block and extracts the text, turning a folder of scans into a searchable catalog without manual re-entry.

Built as the first stage of a scan-to-BIM pipeline for archival engineering documentation.

---

## Results

Evaluated on **49 real archival drawings** (DWG exports, various DPI, mixed quality):

| Method | Mean IoU | Median IoU | Successful (IoU > 0.5) | Wins |
|--------|----------|------------|------------------------|------|
| OpenCV (auto ROI) | **0.655** | **0.962** | **31 / 49** | **38** |
| YOLO (synthetic training) | 0.351 | 0.207 | 16 / 49 | 10 |

**Key finding:** OpenCV with automatic ROI selection based on document orientation significantly outperforms YOLO trained on synthetic data. GOST standardization of stamp position and dimensions makes geometric heuristics more robust than a neural network trained on 25 images.

YOLO failure analysis: the model detects all rectangular stamp-like elements, not just the main title block. Two-class training (main stamp vs. auxiliary elements) is the recommended next step.

See [`docs/article/ARTICLE.md`](docs/article/ARTICLE.md) for the full comparison paper.

---

## How It Works

Three-stage pipeline:

```
Scan → Preprocess → Localize → Extract
        (deskew,     (OpenCV     (EasyOCR)
         auto ROI)    or YOLO)
```

**Preprocessing** — deskew correction, automatic DPI detection, ROI selection based on page orientation (landscape → bottom-right, portrait → bottom, tall portrait → right).

**Localization** — contour-based detection filtered by GOST 2.104 aspect ratios (FORM_3: 3.36, FORM_4: 1.61, FORM_5: 5.40), or YOLOv8n trained on synthetic stamps.

**Extraction** — EasyOCR over the localized region, structured JSON output with per-field bounding boxes and confidence scores.

---

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/Sapar-hub/gost-ocr.git
cd gost-ocr
pip install uv
uv pip sync pyproject.toml
```

---

## Usage

```bash
# Full pipeline (preprocess + localize + OCR)
uv run gost-ocr pipeline /path/to/drawings/ --detector=auto --debug

# Localization only
uv run gost-ocr localize /path/to/drawings/ --detector=opencv

# Reproduce benchmark results
uv run python -m src.gost_ocr.benchmark.compare
```

**Detection methods:** `auto` (YOLO → OpenCV fallback), `opencv`, `yolo`

**ROI options:** `auto` (default), `bottom_right`, `bottom`, `right`, `full_page`, and others

**Output:** JSON per image with `stamp_bbox`, `text_blocks` (text + confidence + coordinates), `full_text`

```json
{
    "stamp_bbox": [1222, 1091, 567, 167],
    "text_blocks": [
        {"text": "Лист", "confidence": 0.999, "box": [[29,15],[83,15],[83,38],[29,38]]}
    ],
    "full_text": "Лист МГТ-2024-ПЗ..."
}
```

---

## Dataset

- **Training:** 25 synthetic images generated per GOST 2.104 (DPI 200/300/400, FORM_3/4/5)
- **Test:** 49 real drawings from open DWG archives, manually labeled with LabelImg
- **Ground truth:** YOLO format, `src/gost_ocr/datasets/labels/test/`
- **Model weights:** `src/gost_ocr/models/yolo/best.pt`

---

## Roadmap

- [ ] Two-class YOLO training: main stamp vs. auxiliary elements
- [ ] Field-level classification (document number, author, date, revision) per GOST 2.104 cell layout
- [ ] Dimension chain recognition (размерные цепочки) for geometry extraction
- [ ] IFC/JSON export for BIM integration

---

## Stack

Python 3.12 · OpenCV · YOLOv8n · EasyOCR · Typer · uv

---

## Paper

[Сравнение методов локализации штампов технической документации ГОСТ: OpenCV vs YOLO](docs/article/ARTICLE.md)
