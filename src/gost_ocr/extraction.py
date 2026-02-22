from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import easyocr
import numpy as np

from .config import DEBUG_EXTRACTION_DIR
from .localization import LocalizationResult

# Initialize EasyOCR model once and reuse it.
_ocr_instance = None


def get_ocr_instance() -> easyocr.Reader:
    """Initializes and returns a singleton easyocr.Reader instance."""
    global _ocr_instance
    if _ocr_instance is None:
        print("Initializing EasyOCR model (ru)...")
        _ocr_instance = easyocr.Reader(["ru"], gpu=False)
        print("EasyOCR model initialized.")
    return _ocr_instance


@dataclass
class TextBlock:
    """Represents a single block of recognized text."""

    text: str
    confidence: float
    box: list[list[int]]  # Box coordinates


@dataclass
class ExtractionResult:
    """Contains all extracted data for a single processed stamp."""

    source_image_path: str
    stamp_bbox: tuple[int, int, int, int]
    text_blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""


def extract_text(
    localization_result: LocalizationResult, debug: bool = False
) -> ExtractionResult | None:
    """
    Extracts text from a localized stamp region using EasyOCR.

    Args:
        localization_result: The result from the localization stage.
        debug: If True, saves the cropped stamp image for debugging.

    Returns:
        An ExtractionResult object containing the extracted text, or None if no stamp was found.
    """
    if not localization_result.stamp:
        return None

    stamp_bbox = localization_result.stamp.bbox
    original_image = localization_result.preprocessed.image

    x, y, w, h = stamp_bbox
    x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

    if w == 0 or h == 0:
        print(
            f"  Warning: Invalid stamp bounding box for {localization_result.preprocessed.original_path.name}. Skipping."
        )
        return None

    stamp_image = original_image[y : y + h, x : x + w]

    if stamp_image.size == 0:
        print(
            f"  Warning: Cropped stamp image is empty for {localization_result.preprocessed.original_path.name}. Skipping."
        )
        return None

    if debug:
        DEBUG_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
        name = localization_result.preprocessed.original_path.stem
        suffix = (
            f"_flip{localization_result.preprocessed.flip_angle}"
            if localization_result.preprocessed.flip_angle != 0
            else ""
        )
        cv2.imwrite(
            str(DEBUG_EXTRACTION_DIR / f"{name}{suffix}_stamp.png"),
            stamp_image,
        )

    ocr = get_ocr_instance()
    # EasyOCR's result is a list of (bbox, text, confidence)
    ocr_result = ocr.readtext(stamp_image)

    text_blocks = []
    full_text_lines = []
    if ocr_result:
        for box, text, confidence in ocr_result:
            # Ensure box is a list of lists of ints for JSON serialization
            int_box = [[int(p[0]), int(p[1])] for p in box]
            text_blocks.append(
                TextBlock(text=text, confidence=float(confidence), box=int_box)
            )
            full_text_lines.append(text)

    return ExtractionResult(
        source_image_path=str(localization_result.preprocessed.original_path),
        stamp_bbox=stamp_bbox,
        text_blocks=text_blocks,
        full_text="\\n".join(full_text_lines),
    )
