import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DEFAULT_IMAGES_PATH = BASE_DIR / "test-all-cv"

FORM_3_ASPECT_RATIO = 185 / 55
FORM_4_ASPECT_RATIO = 185 / 115
FORM_5_ASPECT_RATIO = 297 / 55
FORM_6_ASPECT_RATIO = 185 / 15

FORM_ASPECT_RATIOS = {
    "FORM_3": FORM_3_ASPECT_RATIO,
    "FORM_4": FORM_4_ASPECT_RATIO,
    "FORM_5": FORM_5_ASPECT_RATIO,
    "FORM_6": FORM_6_ASPECT_RATIO,
}

DPI_VALUES = {"200": 200, "300": 300, "400": 400, "600": 600}


def detect_form_type(image_path: str | Path) -> str | None:
    """Detect form type from filename using regex."""
    path_str = str(image_path).upper()
    match = re.search(r"(FORM_[3-6])", path_str)
    if match:
        return match.group(1)
    return None


def get_expected_aspect_ratio(image_path: str | Path) -> float | None:
    """Get expected aspect ratio based on detected form type."""
    form = detect_form_type(image_path)
    if form and form in FORM_ASPECT_RATIOS:
        return FORM_ASPECT_RATIOS[form]
    return None

# --- Contour filtering parameters ---
# These values help filter out false positives during stamp localization.
# They are based on the expected pixel dimensions of a GOST stamp
# scanned at various DPIs (e.g., 200-600 DPI), with a wide tolerance.
# A GOST stamp (185x55mm) at 300 DPI is approx. 2185x650 pixels.
MIN_STAMP_WIDTH_PX = 400  # Filters out small internal table cells and artifacts.
MAX_STAMP_WIDTH_PX = (
    5000  # Filters out contours that are too large (e.g., the whole page border).
)
MIN_STAMP_HEIGHT_PX = 100  # ~55mm at 150 DPI (minimum expected height)
MIN_STAMP_AREA_PX = (
    30000  # Provides an additional filter against small, irrelevant contours.
)


ROI_WIDTH_RATIO = 0.5
ROI_HEIGHT_RATIO = 0.5

DESKEW_MIN_LINE_LENGTH = 100
DESKEW_MAX_LINE_GAP = 10
DESKEW_THRESHOLD = 100

DEFAULT_DEBUG_DIR = Path("debug")
DEBUG_PREPROCESSING_DIR = DEFAULT_DEBUG_DIR / "preprocessing"
DEBUG_PREPROCESSING_ROI_DIR = DEBUG_PREPROCESSING_DIR / "roi"
DEBUG_LOCALIZATION_DIR = DEFAULT_DEBUG_DIR / "localization"
DEBUG_EXTRACTION_DIR = DEFAULT_DEBUG_DIR / "extraction"
OUTPUT_DIR = Path("output")

YOLO_BASE_MODEL = "yolov8n.pt"
YOLO_TRAINED_MODEL = Path(__file__).parent / "models/yolo/best.pt"
YOLO_DATASET_DIR = Path(__file__).parent / "datasets/train"
YOLO_TEST_DIR = Path(__file__).parent / "datasets/test"
