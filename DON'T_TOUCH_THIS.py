import os

import cv2
import typer
import numpy as np

DEBUG = True

app = typer.Typer()


def get_depth(i, hierarchy):
    """Recursively find the depth of a contour."""
    if hierarchy[0][i][3] == -1:
        return 0
    else:
        return 1 + get_depth(hierarchy[0][i][3], hierarchy)


def get_ancestor_at_depth(i, hierarchy, target_depth):
    """
    Find the ancestor of a contour at a specific target depth.
    Returns the index of the ancestor, or the original index if not found.
    """
    current_depth = get_depth(i, hierarchy)
    parent_idx = hierarchy[0][i][3]

    if current_depth <= target_depth or parent_idx == -1:
        return i

    while current_depth > target_depth:
        if parent_idx == -1:
            return i  # Should not happen if depth is > target_depth
        i = parent_idx
        current_depth = get_depth(i, hierarchy)
    return i


def check_gost_stamp_ratio(contour, tolerance=0.15):
    """
    Check if a contour matches the GOST stamp aspect ratio.

    According to GOST 2.104-2006 and GOST Р 21.101-2020:
    - Standard stamp dimensions: 185mm x 55mm
    - Expected aspect ratio: 185/55 ≈ 3.36:1

    Args:
        contour: OpenCV contour
        tolerance: Acceptable deviation from expected ratio (default 15%)

    Returns:
        tuple: (is_valid, aspect_ratio, confidence_score)
    """
    x, y, w, h = cv2.boundingRect(contour)

    if h == 0:
        return False, 0.0, 0.0

    aspect_ratio = w / h
    expected_ratio = 185 / 55  # ≈ 3.3636

    # Calculate how close the ratio is to expected
    ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio

    # Check if within tolerance
    is_valid = ratio_diff <= tolerance

    # Calculate confidence score (1.0 = perfect match, 0.0 = at tolerance boundary)
    confidence_score = max(0.0, 1.0 - (ratio_diff / tolerance))

    return is_valid, aspect_ratio, confidence_score


@app.command()
def visualize_contour_hierarchy(
    image_path: str = typer.Argument(
        default="src/gost_ocr/tests/test_images/test1_half.png",
        help="Path to the image file.",
    ),
):
    """
    Detects the main stamp boundaries and a single union bounding box
    enclosing all its immediate child cells using refined parameters.
    """
    im = cv2.imread(image_path)
    print("Debug state", DEBUG)
    if im is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Image shape: {im.shape}")
    output_im = im.copy()

    if DEBUG:
        os.makedirs("./DEBUG", exist_ok=True)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite("./DEBUG/gray.png", gray)

    # Refined adaptiveThreshold parameters
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )
    if DEBUG:
        cv2.imwrite("./DEBUG/binary_refined.png", binary)

    # Refined findContours method
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Define colors for visualization
    depth0_color = (255, 0, 0)  # Blue for outermost containers (depth 0)
    union_bbox_color = (
        0,
        255,
        0,
    )  # Green for union bounding box of cells (depth 1 children)

    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Reshape for easier access

        # Helper to find depth
        memo = {}

        def get_depth(i):
            if i not in memo:
                parent_idx = hierarchy[i][3]
                if parent_idx == -1:
                    memo[i] = 0
                else:
                    memo[i] = 1 + get_depth(parent_idx)
            return memo[i]

        for i, contour in enumerate(contours):
            # Check if it's a top-level contour (depth 0)
            if get_depth(i) == 0:
                # Check GOST stamp ratio
                is_valid_stamp, aspect_ratio, confidence = check_gost_stamp_ratio(
                    contour
                )

                x0, y0, w0, h0 = cv2.boundingRect(contour)
                cv2.rectangle(output_im, (x0, y0), (x0 + w0, y0 + h0), depth0_color, 2)

                # Print ratio info to console
                print(
                    f"Contour {i}: ratio={aspect_ratio:.2f}, confidence={confidence:.2f}, valid={is_valid_stamp}"
                )

                # Find all its immediate children (depth 1 contours)
                first_child_idx = hierarchy[i][2]
                if first_child_idx != -1:
                    min_x, min_y = float("inf"), float("inf")
                    max_x, max_y = float("-inf"), float("-inf")

                    current_child_idx = first_child_idx
                    while current_child_idx != -1:
                        if get_depth(current_child_idx) == 1:
                            child_contour = contours[current_child_idx]
                            cx, cy, cw, ch = cv2.boundingRect(child_contour)
                            min_x = min(min_x, cx)
                            min_y = min(min_y, cy)
                            max_x = max(max_x, cx + cw)
                            max_y = max(max_y, cy + ch)

                        current_child_idx = hierarchy[current_child_idx][
                            0
                        ]  # Next sibling

                    if min_x != float("inf"):
                        cv2.rectangle(
                            output_im,
                            (min_x, min_y),
                            (max_x, max_y),
                            union_bbox_color,
                            2,
                        )

    if DEBUG:
        cv2.imwrite("./DEBUG/hierarchy_visualization.png", output_im)
    print(
        "Contour hierarchy visualization saved to ./DEBUG/hierarchy_visualization.png"
    )


if __name__ == "__main__":
    app()
