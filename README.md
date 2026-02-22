# GOST-OCR: Automatic Metadata Extraction from Technical Drawing Stamps

This project is a prototype of a Python-based CLI utility for automatically locating and recognizing text within the title block (known as "stamp" or "osnovnaya nadpis") of scanned technical drawings that conform to GOST standards.

## About The Project

In design institutes and archives, the manual processing of digitized drawings is a significant challenge. After scanning, documents are saved with technical filenames, and to catalog them, employees must manually open each file, find the title block, and re-type the metadata (such as project code, sheet number, etc.) into registers. This process is slow, monotonous, and prone to human error.

**GOST-OCR** aims to automate this workflow by providing an image processing pipeline.

### What It Does (Features)

*   **Implements a three-stage pipeline:**
    1.  **Preprocessing:** Corrects image skew and allows for selecting a Region of Interest (ROI) to narrow down the search area.
    2.  **Stamp Localization:** Uses computer vision techniques (contour hierarchy analysis in OpenCV) to locate the title block's frame based on its structural properties.
    3.  **Text Extraction:** Recognizes all text within the located stamp area using the `EasyOCR` engine.
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
    This command will install all required libraries, including `opencv-python`, `typer`, and `easyocr`.

## 🚀 Usage

The utility is run from the command line. The main command is `pipeline`.

### Full Processing Pipeline

This command executes all three stages (preprocess, localize, extract) and saves the results to the `output/` folder.

```bash
gost-ocr pipeline /path/to/your/images/ --debug
```

**Arguments and Options:**

*   `PATH` (required): The path to a single image file or a directory of images.
*   `--roi [POSITION]`: Specifies which part of the image to analyze. This significantly speeds up the search.
    *   **Available positions:** `top`, `bottom`, `left`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right` (default).
*   `--flip` or `-f`: Attempt all rotations (0°, 90°, 180°, 270°). Useful for scans with landscape orientation.
*   `--debug` or `-d`: Enable debug mode. Intermediate processing images will be saved to the `debug/` folder.

### Example

Process all images in the `samples/` directory, searching for the stamp in the bottom-right corner, and save debug files:

```bash
gost-ocr pipeline samples/ --roi bottom_right --debug
```

## 📂 Project Structure

*   `src/gost_ocr/cli.py`: Defines the command-line interface using `Typer`.
*   `src/gost_ocr/preprocessing.py`: Module for image preprocessing (loading, deskewing, ROI cropping).
*   `src/gost_ocr/localization.py`: Module for locating the title block on the image.
*   `src/gost_ocr/extraction.py`: Module for extracting text from the stamp area using `EasyOCR`.
*   `src/gost_ocr/config.py`: Contains project constants and paths.
*   `pyproject.toml`: Project description and dependencies.

## 📄 Output Format

*   **`output/`**: This directory will contain the `.json` files with the recognition results for each processed image.
*   **`debug/`**: If the `--debug` option is enabled, this directory will contain subfolders with intermediate images:
    *   `preprocessing/`: Results of the preprocessing step.
    *   `preprocessing/roi/`: The cropped Regions of Interest (ROI).
    *   `localization/`: Images with bounding boxes of found stamp candidates.
    *   `extraction/`: The cropped stamp images that were sent for OCR.

### Example JSON Output (`<filename>_output.json`)

```json
{
    "source_image_path": "src/gost_ocr/tests/test_images/1.png",
    "stamp_bounding_box": [
        1488,
        1831,
        1226,
        365
    ],
    "text_blocks": [
        {
            "text": "ИЗМ",
            "confidence": 0.999,
            "box": [
                [29, 15],
                [83, 15],
                [83, 38],
                [29, 38]
            ]
        },
        {
            "text": "Лист",
            "confidence": 0.998,
            "box": [
                [96, 15],
                [175, 15],
                [175, 39],
                [96, 39]
            ]
        },
        {
            "text": "МГТ-2024-ПЗ",
            "confidence": 0.85,
            "box": [
                [454, 296],
                [858, 298],
                [858, 336],
                [454, 334]
            ]
        }
    ],
    "full_text": "ИЗМ\nЛист\nМГТ-2024-ПЗ..."
}
```
