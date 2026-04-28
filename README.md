# GOST-OCR: Automatic Metadata Extraction from Technical Drawing Stamps

This project is a Python-based CLI utility for automatically locating and recognizing text within the title block (known as "stamp" or "osnovnaya nadpis") of scanned technical drawings that conform to GOST standards.

<details>
<summary>Spoiler</summary>
  
By the time this project have been created the similair [tool](https://github.com/W24-Service-GmbH/werk24-python) in this specific domain had already been existing. But that tool is just a wrapper API client. The logic and extraction process happens in their cloud.  

</details>

## About The Project

In design institutes and archives, the manual processing of digitized drawings is a significant challenge. After scanning, documents are saved with technical filenames, and to catalog them, employees must manually open each file, find the title block, and re-type the metadata (such as project code, sheet number, etc.) into registers. This process is slow, monotonous, and prone to human error.

**GOST-OCR** aims to automate this workflow by providing an image processing pipeline with two detection methods: OpenCV (traditional) and YOLO (machine learning).

### What It Does (Features)

*   **Implements a three-stage pipeline:**
    1.  **Preprocessing:** Corrects image skew and allows for selecting a Region of Interest (ROI) to narrow down the search area.
    2.  **Stamp Localization:** Uses computer vision (OpenCV contour analysis OR YOLO object detection) to locate the title block.
    3.  **Text Extraction:** Recognizes all text within the located stamp area using the `EasyOCR` engine.
*   **Dual Detection Methods:** Supports both OpenCV (contour-based) and YOLO (deep learning) detection.
*   **Processes Images in Batches:** Works with single image files (`.png`, `.jpg`, `.jpeg`) or an entire folder of images.
*   **Provides Structured Output:** Saves all recognized text blocks, their coordinates, and confidence scores into a structured JSON file for each processed image.
*   **Includes a Debug Mode:** Allows saving intermediate images from each stage for visual inspection and fine-tuning.

### What It Does NOT Do (Limitations)

This prototype is developed with engineering responsibility in mind and has strict boundaries:

| Limitation | Justification |
| :--- | :--- |
| **Only Metadata from the Stamp** | The drawing's geometry (lines, dimensions, schematics) is intentionally not analyzed. This task requires certified software and an engineer's expertise. |
| **Not a Human Replacement** | The tool is designed to automate routine tasks, not to make final decisions. Results, especially those with low confidence, require human verification. |
| **Not for BIM Integration** | The extracted data is not intended for the automatic generation of 3D models or other critical documentation without manual review. |


## Installation

This project requires Python 3.12+.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sapar-hub/gost-ocr.git
    cd gost-ocr
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project uses `uv` for dependency management.
    ```bash
    pip install uv
    uv pip sync pyproject.toml
    ```
    This command will install all required libraries, including `opencv-python`, `typer`, `easyocr`, and `ultralytics`.


## Usage

The utility is run from the command line using `uv run gost-ocr <COMMAND> [OPTIONS]`.

### Detection Methods

Use `--detector` to choose the detection method:
- `auto` (default): Try YOLO, fallback to OpenCV
- `yolo`: Use YOLO detection only
- `opencv`: Use OpenCV contour detection

### Full Processing Pipeline (`pipeline`)

This command executes all three stages (preprocess, localize, extract) and saves the results to the `output/` folder.

```bash
uv run gost-ocr pipeline /path/to/your/images/ --detector=auto --debug
```

### YOLO Detection Only (`detect`)

Standalone YOLO detection without OCR - useful for testing detection accuracy.

```bash
uv run gost-ocr detect /path/to/your/images/
```

### Localization Only (`localize`)

This command performs preprocessing and stamp localization without text extraction.

```bash
uv run gost-ocr localize /path/to/your/images/ --detector=yolo --debug
```

### Evaluation (`evaluate`)

This command assesses the quality of the OCR pipeline against ground truth data.

```bash
uv run gost-ocr evaluate <GROUND_TRUTH_DIR> --output-dir <OUTPUT_DIR>
```

### Common Options

*   `--roi [POSITION]`: Specifies which part of the image to analyze. Available: `top`, `bottom`, `left`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right`, `full_page`, `corners`, `auto` (default).
*   `--dpi [VALUE]`: Specifies the DPI of the image. Available: `auto` (default), `200`, `300`, `400`, `600`.
*   `--flip` or `-f`: Attempt all rotations (0°, 90°, 180°, 270°) during preprocessing.
*   `--debug` or `-d`: Enable debug mode. Save intermediate images to `debug/`.

## Project Structure

```
src/gost_ocr/
├── cli.py                      # Typer CLI entrypoint
├── config.py                  # Configuration and constants
├── preprocessing.py           # Image preprocessing (deskew, ROI, flip)
├── detection/                 # Unified detection interface
│   ├── factory.py            # get_detector() factory
│   ├── yolo.py              # YOLO detector
│   └── opencv.py            # OpenCV detector
├── localization.py         # OpenCV detection (legacy)
├── extraction.py            # EasyOCR text extraction
├── evaluation.py            # Metrics (IoU, CER, WER)
├── models/yolo/
│   └── best.pt             # Trained YOLO weights
├── datasets/
│   ├── train/             # Training synthetic data
│   └── test/              # Test images + GT labels
└── tests/
    ├── analyze_comparison.py
    └── generate_visualization.py
```

## Evaluation Results (YOLO vs OpenCV)

The system was evaluated on 11 real archive drawings, comparing YOLO and OpenCV detection methods.

### Dataset
- **Training:** 25 synthetic images (DPI 200-400, FORM_3/4/5)
- **Test:** 11 real archive drawings
- **Ground Truth:** `src/gost_ocr/datasets/test/labels/`

### Aggregate Metrics

| Metric | YOLO | OpenCV |
|--------|-----|-------|
| Mean IoU | **0.426** | 0.361 |
| Median IoU | **0.865** | 0.328 |
| Wins | **6** | 1 |
| Ties | 4 | 4 |
| Full detections (IoU > 0.5) | 5 | 3 |
| Complete failures (IoU = 0) | 5 | 6 |

### Key Findings

1. **YOLO outperforms OpenCV** - Winner 6:1 with higher mean and median IoU
2. **Transfer learning works** - Training on 25 synthetic images generalizes to real scans
3. **YOLO more stable** - Fewer complete failures (5 vs 6)
4. **Hybrid approach possible** - Use `auto` mode for best results

## Output Format

*   **`output/`**: Contains `.json` files with recognition results.
*   **`output/*_yolo.json`**: YOLO detection results.
*   **`output/*_output.json`**: OpenCV detection + OCR results.
*   **`debug/`**: If `--debug` is enabled, intermediate processing images.

### Example JSON Output

```json
{
    "source_image_path": "src/gost_ocr/datasets/test/images/uzly-1-2-3-4-5.jpg",
    "stamp_bbox": [1222, 1091, 567, 167],
    "text_blocks": [
        {
            "text": "Лист",
            "confidence": 0.999,
            "box": [[29, 15], [83, 15], [83, 38], [29, 38]]
        }
    ],
    "full_text": "Лист МГТ-2024-ПЗ..."
}
```

## Usage Preview

<video src="https://github.com/user-attachments/assets/b1913a1a-7479-4f1f-ba1a-ba56507206c2" autoplay muted playsinline controls width="600" height="360" >
    Your browser does not support the video tag.
</video>

<video src="https://github.com/user-attachments/assets/95f0faad-95b4-4f8a-b147-ff2935fbba96" autoplay muted playsinline controls width="600" height="360">    
</video>