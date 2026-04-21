# GOST-OCR: Automatic Metadata Extraction from Technical Drawing Stamps

This project is a prototype of a Python-based CLI utility for automatically locating and recognizing text within the title block (known as "stamp" or "osnovnaya nadpis") of scanned technical drawings that conform to GOST standards.

<details>
  <summary>Spoiler</summary>
  
  By the time this project have been created the similair [tool](https://github.com/W24-Service-GmbH/werk24-python) in this specific domain had already been existing. But that tool is just a wrapper API client. The logic and extraction process happens in their cloud.  
  
</details>

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

## Usage

The utility is run from the command line using `uv run gost-ocr <COMMAND> [OPTIONS]`.

### Full Processing Pipeline (`pipeline`)

This command executes all three stages (preprocess, localize, extract) and saves the results to the `output/` folder.

```bash
uv run gost-ocr pipeline /path/to/your/images/ --debug
```

**Arguments and Options:**

*   `PATH` (required): The path to a single image file or a directory of images.
*   `--roi [POSITION]`: Specifies which part of the image to analyze for the stamp. This significantly narrows down the search area.
    *   **Available positions:** `top`, `bottom`, `left`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right`, `full_page`, `corners`, `auto` (default).
*   `--dpi [VALUE]`: Specifies the DPI of the image.
    *   **Available values:** `auto` (default), `200`, `300`, `400`, `600`. The `auto` option attempts to detect the DPI automatically.
*   `--filter-by-size/--no-filter-by-size`: Controls whether detected stamps are filtered by pre-configured size thresholds.
    *   **Default:** `False` (stamps are NOT filtered by size by default).
*   `--flip` or `-f`: Attempt all rotations (0°, 90°, 180°, 270°) during preprocessing. Useful for scans with varying orientations.
*   `--debug` or `-d`: Enable debug mode. Intermediate processing images (preprocessing, localization) will be saved to the `debug/` folder. In this mode, extracted text with confidence will also be saved as `.txt` files in `debug/extraction/`.

### Localization Only (`localize`)

This command performs preprocessing and stamp localization without text extraction. It's useful for testing and tuning the localization stage.

```bash
uv run gost-ocr localize /path/to/your/images/ --debug
```
**Arguments and Options:** Same as `pipeline` command.

### Evaluation (`evaluate`)

This command assesses the quality of the OCR pipeline against ground truth data.

**Note:** Before running `evaluate`, you must execute the `pipeline` command on your ground truth images to generate the necessary OCR output files.

```bash
uv run gost-ocr evaluate <GROUND_TRUTH_DIR> --output-dir <OUTPUT_DIR>
```

**Arguments and Options:**

*   `GROUND_TRUTH_DIR` (required): Path to the directory containing ground truth JSON files.
*   `--output-dir` or `-o`: Path to the directory containing OCR results (JSON files).
*   `--iou-threshold`: Minimum IoU for successful localization (default: 0.5).
*   `--cer-threshold`: Maximum CER for successful text recognition (default: 0.1).
*   `--wer-threshold`: Maximum WER for successful word recognition (default: 0.2).
*   `--save-report` or `-s`: Save the evaluation report to a JSON file.

### Example

Process all images in the `samples/` directory, searching for the stamp in the bottom-right corner, and save debug files:

```bash
uv run gost-ocr pipeline samples/ --roi bottom_right --debug
```

## Project Structure

*   `src/gost_ocr/cli.py`: Defines the command-line interface using `Typer`.
*   `src/gost_ocr/preprocessing.py`: Module for image preprocessing (loading, deskewing, ROI cropping).
*   `src/gost_ocr/localization.py`: Module for locating the title block on the image.
*   `src/gost_ocr/extraction.py`: Module for extracting text from the stamp area using `EasyOCR`.
*   `src/gost_ocr/config.py`: Contains project constants and paths.
*   `pyproject.toml`: Project description and dependencies.

## Output Format

*   **`output/`**: This directory will contain the `.json` files with the recognition results for each processed image.
*   **`debug/`**: If the `--debug` option is enabled, this directory will contain subfolders with intermediate images and text output:
    *   `preprocessing/`: Results of the preprocessing step.
    *   `preprocessing/roi/`: The cropped Regions of Interest (ROI).
    *   `localization/`: Images with bounding boxes of found stamp candidates.
    *   `extraction/`: Contains `.txt` files with extracted text and confidence scores for the most confident stamp per image. Cropped stamp images are no longer saved here.

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
    "full_text": "ИЗМ Лист МГТ-2024-ПЗ..."
}
```

## Evaluation Results

The system was evaluated on a dataset of 25 synthetic images with varying DPI and GOST stamp forms (FORM_3, FORM_4, FORM_5).

*   **Dataset (Ground Truth):** `src/gost_ocr/tests/ground_truth/`
*   **Evaluation Output:** `output/`

### Aggregate Metrics

| Metric                           | Value       |
| :------------------------------- | :---------- |
| Images processed                 | 25          |
| Localization success (IoU ≥ 0.5) | 23/25 (92%) |
| Text recognition success         | 0/25 (0%)   |
| Mean IoU                         | 0.86        |
| Median IoU                       | 0.99        |
| Mean CER                         | 0.67        |
| Mean WER                         | 1.47        |

### Interpretation

The evaluation indicates that the system is highly effective at localizing GOST stamps, achieving a 92% success rate and high IoU values. This robust localization forms a strong foundation for the overall automation pipeline. However, the current text recognition component exhibits low accuracy (0% success rate), with high Character Error Rate (CER) and Word Error Rate (WER). This suggests that while stamps are correctly identified, the extracted text often contains errors, necessitating human verification for reliable metadata extraction. Further improvements are required for the OCR stage to achieve fully autonomous operation.

## Usage preview

<video src="https://github.com/user-attachments/assets/b1913a1a-7479-4f1f-ba1a-ba56507206c2" autoplay muted playsinline controls width="600" height="360" >
    Your browser does not support the video tag.
</video>

<video src="https://github.com/user-attachments/assets/95f0faad-95b4-4f8a-b147-ff2935fbba96" autoplay muted playsinline controls width="600" height="360">    
</video>
