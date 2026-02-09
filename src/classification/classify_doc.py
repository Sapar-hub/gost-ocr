import math
import cv2
import numpy as np
from src.config import FORM_3_5_ASPECT_RATIO, FORM_4_6_ASPECT_RATIO, ASPECT_RATIO_TOLERANCE
from src.localization.locate_stamp import count_peaks, is_left_aligned

# CLASSIFICATION OF DRAWING
def classify_document_type(stamp_image, zone):
    """
    Analyze the structure of table
    """
    if stamp_image.shape[0] == 0 or stamp_image.shape[1] == 0:
        return "UNKNOWN_INVALID_IMAGE"

    h, w = stamp_image.shape[:2]
    aspect_ratio = w / h

    # Convert stamp_image to grayscale if it's not already
    gray_stamp_image = cv2.cvtColor(stamp_image, cv2.COLOR_BGR2GRAY) if len(stamp_image.shape) == 3 else stamp_image

    vertical_proj = np.sum(gray_stamp_image, axis=0)
    cell_count = count_peaks(vertical_proj)

    if cell_count == 6 and abs(aspect_ratio - FORM_3_5_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE:
        return "FORM_1" # DRAWING
    elif cell_count == 4 and abs(aspect_ratio - FORM_4_6_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE and is_left_aligned(zone):
        return "FORM_6" # SPECIFICATION OF DRAWING
    elif cell_count == 4 and abs(aspect_ratio - FORM_4_6_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE:
        return "FORM_4_6_SUBSEQUENT_PAGE" # Subsequent page, not necessarily left aligned
    else:
        return "UNKNOWN"
    # NUANCE: Classification is based on geometry of GOST's cells which is persistent
    # so that no need for low-resolution heuristic text dependency is needed 
