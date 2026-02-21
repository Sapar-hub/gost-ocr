# Technical Specifications: GOST-compliant Drawing OCR

This document outlines the technical specifications for a Python application designed to perform Optical Character Recognition (OCR) on GOST-compliant technical drawings. The goal is to extract structured data from the title block (stamp).

## 1. Core Functionality

The primary objective of this project is to automatically locate the title block in a scanned technical drawing, segment it into individual cells, and extract the text from each cell.

## 2. Project Architecture and Data Flow

The application follows a two-stage pipeline architecture (extraction stage planned for future implementation).

**Data Flow:**

1. **Input:** Image files (PNG, JPG) from a folder or single file path.
2. **Preprocessing:** Deskew correction, optional flip rotations, and ROI extraction.
3. **Localization:** Contour detection with hierarchy analysis to find the stamp bounding box.
4. **Extraction (TODO):** OCR will be performed using PaddleOCR.

## 3. Input/Output

* **Input:**
    * Format: Image files (PNG, JPG, JPEG).
    * Source: Path to a local file or directory.
    * Default path: `src/gost_ocr/tests/test_images/`
* **Output:**
    * **Primary:** Stamp bounding box coordinates and metadata.
    * **Debug output:** Intermediate images saved to `debug/preprocessing/` and `debug/localization/`.

## 4. Key Modules

* `cli.py`: Command-line interface using Typer with commands: `preprocess`, `localize`, `pipeline`.
* `preprocessing.py`: Image loading, deskew, flip rotations, ROI extraction.
* `localization.py`: Contour detection with RETR_TREE hierarchy, GOST ratio validation.
* `config.py`: Configuration constants (GOST dimensions, thresholds, paths).

## 5. Stage 1: Preprocessing

### 5.1. Image Loading

* Loads PNG/JPG/JPEG files from specified path.
* Supports single file or directory input.
* Outputs count of loaded files to console.

### 5.2. Deskew

* Uses Hough Line Transform (`cv2.HoughLinesP`) to detect dominant line angles.
* Filters near-horizontal lines (±45° from horizontal).
* Calculates histogram peak angle and rotates image to correct skew.

### 5.3. Flip Rotations

* Disabled by default.
* Enabled via `--flip` / `-f` CLI flag.
* Applies rotations: 0°, 90°, 180°, 270° (generates 4 versions per image).

### 5.4. ROI Extraction

* Extracts 1/4 of image area (50% width × 50% height).
* Default position: bottom-right corner.
* Configurable via `--roi` flag: `bottom_right`, `bottom_left`, `top_right`, `top_left`.

### 5.5. Debug Output

When `--debug` / `-d` flag is set:
* Saves to `debug/preprocessing/`
* Files: `{name}_preprocessed.png`, `{name}_roi.png`

## 6. Stage 2: Localization

### 6.1. Contour Detection

* Converts ROI to grayscale.
* Applies adaptive thresholding (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`).
* Morphological closing with 25×25 kernel to form solid regions.
* Uses `cv2.findContours` with `RETR_TREE` to get hierarchy.

### 6.2. Hierarchy Analysis

* `RETR_TREE` returns parent-child relationships.
* The algorithm searches for contours that are parents (have child contours) but do not have a parent themselves.
* This correctly identifies the outer frame of the stamp, ignoring the inner cells.

### 6.3. Rectangular Contour Filtering

* Uses `cv2.approxPolyDP` to check if contour has 4 vertices.
* Filters by minimum area, width, and height.

### 6.4. GOST Ratio Validation

Validates aspect ratio against GOST standards:
* **FORM_3_5**: 185×55mm → aspect ratio ≈ 3.36
* **FORM_4_6**: 185×15mm → aspect ratio ≈ 12.33
* Tolerance: ±0.5

### 6.5. Draw All Mode

When `--draw-all` flag is set:
* Skips GOST ratio validation.
* Finds all rectangular contours.
* Draws all contours on debug image.
* Outputs count of found contours instead of stamp info.

### 6.6. Debug Output

When `--debug` / `-d` flag is set:
* Saves to `debug/localization/`
* File: `{name}_localization.png` with bounding boxes drawn

## 7. CLI Commands

```
gost-ocr --help
gost-ocr preprocess [PATH] [--flip] [--roi POSITION] [--debug]
gost-ocr localize [PATH] [--flip] [--draw-all] [--roi POSITION] [--debug]
gost-ocr pipeline [PATH] [OPTIONS]  # alias for localize
```

**Options:**
* `--flip` / `-f`: Apply all rotations (0/90/180/270°)
* `--draw-all`: Show all rectangular contours (skip GOST validation)
* `--roi`: ROI position (bottom_right, bottom_left, top_right, top_left)
* `--debug` / `-d`: Save intermediate results

## 8. Dependencies

* `opencv-python`: Computer vision (contours, transformations, thresholding)
* `numpy`: Numerical operations
* `typer`: CLI framework
* `PaddleOCR`: (planned for extraction stage)

## 9. Future Work

* **Stage 3: Extraction** - Implement PaddleOCR for text extraction from stamp cells.
* **Segmentation** - Divide stamp into individual cells based on GOST grid structure.
* **Batch processing** - Process multiple drawings with progress reporting.
* **JSON output** - Structured output format with confidence scores.
