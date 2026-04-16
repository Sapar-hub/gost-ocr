from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_IMAGES_PATH = BASE_DIR / "tests" / "test_images"

FORM_3_5_ASPECT_RATIO = 185 / 55
FORM_4_6_ASPECT_RATIO = 185 / 15

# --- Contour filtering parameters ---
# These values help filter out false positives during stamp localization.
# They are based on the expected pixel dimensions of a GOST stamp
# scanned at various DPIs (e.g., 200-600 DPI), with a wide tolerance.
# A GOST stamp (185x55mm) at 300 DPI is approx. 2185x650 pixels.
MIN_STAMP_WIDTH_PX = 1000  # Filters out small internal table cells and artifacts.
MAX_STAMP_WIDTH_PX = (
    5000  # Filters out contours that are too large (e.g., the whole page border).
)
MIN_STAMP_HEIGHT_PX = 250  # ~55mm at 150 DPI (minimum expected height)
MIN_STAMP_AREA_PX = (
    50000  # Provides an additional filter against small, irrelevant contours.
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
