import cv2
import numpy as np
import math
from src.localization.locate_stamp import locate_stamp_candidates
from src.classification.classify_doc import classify_document_type
from src.extraction.extract_meta import extract_metadata # Adding extraction for completeness
from src.config import FORM_3_5_ASPECT_RATIO, FORM_4_6_ASPECT_RATIO, ASPECT_RATIO_TOLERANCE


def rotate_image(image, angle):
    """Rotates an image around its center."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def deskew_image(image):
    """
    Detects and corrects the skew of an image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Invert colors for better line detection if background is dark (common in scanned docs)
    # gray = cv2.bitwise_not(gray)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("No lines detected for deskewing.")
        return image

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Avoid vertical lines as they don't contribute to skew
        if x2 - x1 != 0:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Filter out near-vertical lines, only consider near-horizontal lines
            if abs(angle) < 45 or abs(angle) > 135: # lines that are mostly horizontal
                angles.append(angle)

    if not angles:
        print("No suitable lines found for skew detection.")
        return image

    # Average the angles to find the overall skew angle
    # We should normalize angles to be within a -90 to 90 range, typically -45 to 45 for deskew
    hist = np.histogram(angles, bins=180) # Bins from -90 to 90 degrees
    peak_angle = hist[1][np.argmax(hist[0])]

    # Adjust the angle for rotation (OpenCV rotates counter-clockwise for positive angles)
    # If the text is skewed clockwise, the angle will be negative, so we rotate clockwise (positive angle)
    # If the text is skewed counter-clockwise, the angle will be positive, so we rotate counter-clockwise (negative angle)
    skew_angle = peak_angle
    print(f"Detected skew angle: {skew_angle:.2f} degrees")

    # Rotate the image to correct the skew
    deskewed_image = rotate_image(image, skew_angle)
    return deskewed_image

def main():
    print("Starting metadata extraction process simulation...")

    # 1. Load an actual image for demonstration
    # Ensure test_image.png exists in the gost-2.104-101-ocr directory
    image_path = 'test_image.png' # Using a placeholder for now
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{image_path}' exists for testing.")
        # Fallback to dummy image if actual image is not found
        original_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # 2. Apply global image preprocessing (deskewing and rotation)
    preprocessed_image = deskew_image(original_image)
    
    print("\nAttempting to locate stamp candidates...")
    candidates = locate_stamp_candidates(preprocessed_image) # Pass preprocessed image

    if candidates:
        print(f"Found {len(candidates)} potential stamp candidates.")
        # For simplicity, take the first candidate
        best_candidate_zone, best_candidate_stamp_image = candidates[0]
        print(f"Best candidate found in zone: {best_candidate_zone}")

        print("\nAttempting to classify document type...")
        doc_type = classify_document_type(best_candidate_stamp_image, best_candidate_zone)
        print(f"Classified document type: {doc_type}")

        print("\nAttempting to extract metadata...")
        metadata = extract_metadata(best_candidate_stamp_image, doc_type)
        print("Extracted Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
    else:
        print("No stamp candidates found. (This might be expected for a simple test image or if preprocessing was insufficient).")
        # To test classification and extraction, let's create a dummy stamp image
        print("\nSimulating a dummy stamp image for classification and extraction test:")
        simulated_stamp = np.zeros((100, 400, 3), dtype=np.uint8) # A black rectangle
        simulated_zone = (0.85, 0.70) # Assume bottom-right

        doc_type_simulated = classify_document_type(simulated_stamp, simulated_zone)
        print(f"Simulated Classified document type: {doc_type_simulated}")

        metadata_simulated = extract_metadata(simulated_stamp, doc_type_simulated)
        print("Simulated Extracted Metadata:")
        for key, value in metadata_simulated.items():
            print(f"  {key}: {value}")
        

    print("\nProcess simulation complete.")

if __name__ == "__main__":
    main()
