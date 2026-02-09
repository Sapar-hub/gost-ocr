import math
import re
import cv2
import numpy as np

# EXTRACTION
from ..segmentation.segment_cells import segment_cells as segment_cells_impl

def segment_stamp_cells(stamp_image, doc_type):
    """
    Wrapper function that calls the robust cell segmentation logic.
    """
    return segment_cells_impl(stamp_image, doc_type)

import easyocr

# Initialize the OCR reader once. This is important for performance.
# It will download the models the first time it's run.
# Using both Russian and English as GOST documents often contain both.
reader = easyocr.Reader(['ru', 'en'])

def enhance_cell_image(cell_image):
    """
    Applies a series of image enhancement techniques to improve OCR accuracy.
    """
    if cell_image is None or cell_image.size == 0:
        return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY) if len(cell_image.shape) == 3 else cell_image

    # 2. Binarization using adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # 3. Denoising
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

    return denoised

def ocr(cell_image):
    """
    Performs OCR on a cell image using EasyOCR.
    """
    # Apply image enhancement before OCR
    enhanced_image = enhance_cell_image(cell_image)

    if enhanced_image is None:
        return "" # Return empty string if enhancement fails

    # Use EasyOCR to read text from the image
    # The result is a list of (bounding_box, text, confidence)
    results = reader.readtext(enhanced_image)
    
    # Concatenate all found text fragments into a single string
    recognized_text = " ".join([res[1] for res in results])
    
    return recognized_text

def validate_with_regex(text, pattern):
    """
    Validates text against a regex pattern.
    Returns the matched text if valid, otherwise None.
    """
    if text is None:
        return None
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None

def tesseract_confidence(cell_image):
    """
    Simulates Tesseract's confidence score for a recognized cell.
    In a real system, this would come from the OCR engine.
    """
    # Placeholder: return a high confidence for now.
    return 0.98

def geometry_validation(cell_image):
    """
    Placeholder for geometry validation of a cell (e.g., checking if text bounding box
    is within expected cell boundaries).
    """
    # For now, assume perfect geometric validation.
    return 0.95

def extract_metadata(stamp_image, doc_type):
    # Сегментация ячеек по координатам ГОСТ
    cells = segment_stamp_cells(stamp_image, doc_type)

    # Извлечение с валидацией
    metadata = {}
    
    # Placeholder for actual cells for code and sheet
    code_cell_image = cells.get('code')
    sheet_cell_image = cells.get('sheet')

    if code_cell_image is not None:
        metadata['project_code'] = validate_with_regex(
            ocr(code_cell_image), 
            r'[МM]-\d{3,4}(-[А-Я]{2,3})?'  # допускаем отсутствие раздела
        )
    else:
        metadata['project_code'] = None

    if sheet_cell_image is not None:
        metadata['sheet'] = validate_with_regex(
            ocr(sheet_cell_image), 
            r'(\d{1,2})/(\d{1,2})'  # 07/24
        )
    else:
        metadata['sheet'] = None

    # Confidence scoring
    metadata['confidence'] = min(
        tesseract_confidence(code_cell_image), # Assuming code_cell is critical for confidence
        geometry_validation(code_cell_image)  # проверка расположения ячейки
    )

    # Пометка на ручную проверку при низком доверии
    if metadata['confidence'] < 0.85:
        metadata['requires_review'] = True
    else:
        metadata['requires_review'] = False

    return metadata

