import cv2
import numpy as np

def segment_cells(stamp_image, doc_type):
    """
    Segments the stamp image into individual cells using Hough Line Transform to detect grid lines.
    """
    if stamp_image is None or stamp_image.size == 0:
        return {}

    # 1. Preprocessing
    gray = cv2.cvtColor(stamp_image, cv2.COLOR_BGR2GRAY) if len(stamp_image.shape) == 3 else stamp_image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 2. Line Detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is None:
        print("Warning: No lines detected in stamp. Falling back to basic segmentation.")
        return _fallback_segmentation(stamp_image, doc_type)

    # 3. Process and Filter Lines
    horizontal_lines, vertical_lines = _process_lines(lines, stamp_image.shape)
    
    if not horizontal_lines or not vertical_lines:
        print("Warning: Could not determine grid structure. Falling back to basic segmentation.")
        return _fallback_segmentation(stamp_image, doc_type)

    # 4. Extract Cells from Grid
    cells = _extract_cells_from_grid(stamp_image, horizontal_lines, vertical_lines)

    # 5. Map cells to names based on doc_type (placeholder, requires GOST layout knowledge)
    # For now, we'll just return the grid of cells. A real implementation would map these.
    # For example, cells[0][0] might be the 'code' for FORM_1.
    
    # A simple mapping for demonstration, assuming a 2x2 grid for simplicity
    named_cells = {}
    if doc_type == "FORM_1" and len(cells) > 0 and len(cells[0]) > 0:
        named_cells['code'] = cells[0][0] # Top-left
        if len(cells) > 1 and len(cells[1]) > 1:
            named_cells['sheet'] = cells[1][1] # Bottom-right
            
    # If no specific mapping, return the first cell found as 'unknown'
    if not named_cells and cells:
        named_cells['unknown_cell_1'] = cells[0][0]

    return named_cells

def _process_lines(lines, shape):
    """Helper to classify lines as horizontal or vertical and merge them."""
    h, w = shape[:2]
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        if angle < 10 or angle > 170: # Horizontal
            horizontal_lines.append(y1)
        elif angle > 80 and angle < 100: # Vertical
            vertical_lines.append(x1)

    # Merge close lines (simple averaging)
    def merge_lines(lines, threshold=10):
        lines.sort()
        merged = []
        if not lines:
            return []
        current_group = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - current_group[-1] < threshold:
                current_group.append(lines[i])
            else:
                merged.append(int(np.mean(current_group)))
                current_group = [lines[i]]
        merged.append(int(np.mean(current_group)))
        return merged

    horizontal_lines = merge_lines(horizontal_lines)
    vertical_lines = merge_lines(vertical_lines)

    # Add image borders to complete the grid
    if 0 not in horizontal_lines: horizontal_lines.insert(0, 0)
    if h not in horizontal_lines: horizontal_lines.append(h)
    if 0 not in vertical_lines: vertical_lines.insert(0, 0)
    if w not in vertical_lines: vertical_lines.append(w)
    
    return horizontal_lines, vertical_lines

def _extract_cells_from_grid(image, h_lines, v_lines):
    """Extracts cell images based on the detected grid lines."""
    cells = []
    for i in range(len(h_lines) - 1):
        row = []
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i+1]
            x1, x2 = v_lines[j], v_lines[j+1]
            cell_image = image[y1:y2, x1:x2]
            if cell_image.size > 0: # Ensure cell is not empty
                row.append(cell_image)
        if row:
            cells.append(row)
    return cells

def _fallback_segmentation(stamp_image, doc_type):
    """A basic segmentation to be used when line detection fails."""
    cells = {}
    h, w = stamp_image.shape[:2]
    if doc_type == "FORM_1":
        cells['code'] = stamp_image[int(0.1*h):int(0.3*h), int(0.05*w):int(0.4*w)]
        cells['sheet'] = stamp_image[int(0.7*h):int(0.9*h), int(0.5*w):int(0.7*w)]
    else: # Default
        cells['code'] = stamp_image[int(0.1*h):int(0.3*h), int(0.1*w):int(0.5*w)]
    return cells
