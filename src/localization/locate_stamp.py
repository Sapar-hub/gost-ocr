import cv2
import math
import numpy as np
from src.config import FORM_3_5_ASPECT_RATIO, FORM_4_6_ASPECT_RATIO, ASPECT_RATIO_TOLERANCE

# LOCALIZATION
def extract_corner_zone(image, zone_coords, zone_size_ratio=0.3):
    """
    Extracts a corner zone from the image.
    zone_coords: (x_ratio, y_ratio) indicating the starting point of the zone (e.g., (0.85, 0.70) for bottom-rightish)
    zone_size_ratio: Determines the size of the corner zone relative to the image (e.g., 0.3 means 30% of width/height).
    # NOTE: In a real implementation, a preprocessing step to correct for image rotation/skew
    # would be applied to the 'image' before this function is called.
    """
    h, w = image.shape[:2]
    zone_w = int(w * zone_size_ratio)
    zone_h = int(h * zone_size_ratio)

    # Calculate actual starting points for the crop to ensure it's in the specified corner
    if zone_coords[0] > 0.5: # right side
        x1 = w - zone_w
    else: # left side
        x1 = 0

    if zone_coords[1] > 0.5: # bottom side
        y1 = h - zone_h
    else: # top side
        y1 = 0
    
    x2 = x1 + zone_w
    y2 = y1 + zone_h

    return image[y1:y2, x1:x2]


def find_rectangular_contours(roi):
    """
    Finds rectangular contours within a given Region of Interest (ROI).
    """
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    rect_contours = []
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the contour has 4 vertices, it might be a rectangle
        if len(approx) == 4:
            # Get bounding box and aspect ratio
            x, y, w_c, h_c = cv2.boundingRect(approx)
            
            # Filter based on reasonable size for a stamp (adjust as needed)
            if 50 < w_c < roi.shape[1] * 0.9 and 20 < h_c < roi.shape[0] * 0.9: # arbitrary size check
                rect_contours.append(approx)
    return rect_contours

def is_stamp_proportions(contour):
    """
    Checks whether the cropped contour matches the GOST stamp proportions.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0: # Avoid division by zero
        return False
    aspect_ratio = w / float(h)

    # Check for Form 3/5 proportions (approx 3.36:1)
    is_form_3_5 = abs(aspect_ratio - FORM_3_5_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE

    # Check for Form 4/6 proportions (approx 12.33:1)
    is_form_4_6 = abs(aspect_ratio - FORM_4_6_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE

    return is_form_3_5 or is_form_4_6

def priority_score(zone_coords):
    """
    Assigns a priority score to a zone based on its location.
    Bottom-right (0.85, 0.70) should have the highest score.
    """
    # Simple scoring: higher values for bottom-right
    if zone_coords == (0.85, 0.70): # Bottom-right
        return 4
    elif zone_coords == (0.85, 0.0): # Top-right
        return 3
    elif zone_coords == (0.0, 0.70): # Bottom-left
        return 2
    elif zone_coords == (0.0, 0.0): # Top-left
        return 1
    return 0 # Should not happen with defined zones

def locate_stamp_candidates(image):
    """
    Localization of GOST stamp in four corners of image
    """
    candidates = []
    # Zones are defined as (x_start_ratio, y_start_ratio)
    zones = [(0.85, 0.70), (0.85, 0.0), (0.0, 0.70), (0.0, 0.0)]

    for zone in zones:
        roi = extract_corner_zone(image, zone)
        contours = find_rectangular_contours(roi)

        for contour in contours:
            if is_stamp_proportions(contour):
                # Adjust contour coordinates back to original image frame
                x_roi, y_roi, w_c, h_c = cv2.boundingRect(contour) # Bounding rect of the contour in ROI
                h, w = image.shape[:2]
                zone_w = int(w * 0.3) # Assuming default zone_size_ratio
                zone_h = int(h * 0.3)
                
                x_offset = 0
                y_offset = 0

                if zone[0] > 0.5: # right side
                    x_offset = w - zone_w
                
                if zone[1] > 0.5: # bottom side
                    y_offset = h - zone_h
                
                # Get the bounding box in original image coordinates
                x_abs = x_offset + x_roi
                y_abs = y_offset + y_roi
                w_abs = w_c
                h_abs = h_c

                cropped_stamp_image = image[y_abs:y_abs+h_abs, x_abs:x_abs+w_abs]
                candidates.append((zone, cropped_stamp_image))

    return sorted(candidates, key=lambda x: priority_score(x[0]), reverse=True) # Higher score = higher priority

def count_peaks(projection, min_peak_height_ratio=0.1, min_peak_distance=10):
    """
    Counts significant peaks in a 1D projection array.
    projection: 1D array representing sum of pixel intensities.
    min_peak_height_ratio: Minimum height of a peak relative to the max projection value.
    min_peak_distance: Minimum distance between two peaks.
    """
    if len(projection) == 0:
        return 0

    peaks = 0
    max_val = np.max(projection)
    threshold = max_val * min_peak_height_ratio
    
    in_peak = False
    for i in range(len(projection)):
        if projection[i] > threshold:
            if not in_peak:
                peaks += 1
                in_peak = True
        else:
            in_peak = False
    return peaks

def is_left_aligned(zone_coords):
    """
    Checks if the zone corresponds to a left-aligned position (e.g., bottom-left or top-left).
    """
    return zone_coords[0] < 0.5 # If x_ratio is less than 0.5, it's on the left side
