from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DEBUG_PREPROCESSING_DIR,
    DEBUG_PREPROCESSING_ROI_DIR,
    DESKEW_MAX_LINE_GAP,
    DESKEW_MIN_LINE_LENGTH,
    DESKEW_THRESHOLD,
    ROI_HEIGHT_RATIO,
    ROI_WIDTH_RATIO,
)


A4_SIZES = {
    150: (1754, 2480),
    200: (1754, 2480),
    300: (2480, 3508),
    400: (3508, 4961),
    600: (4961, 7016),
}
A3_SIZES = {
    150: (2480, 3508),
    200: (2339, 3307),
    300: (3508, 4961),
    400: (4961, 7016),
    600: (7016, 9933),
}


def detect_dpi(image: np.ndarray) -> int:
    """Detect DPI from image dimensions."""
    h, w = image.shape[:2]

    for dpi, (expected_w, expected_h) in A4_SIZES.items():
        if abs(w - expected_w) < 50 and abs(h - expected_h) < 50:
            return dpi

    for dpi, (expected_w, expected_h) in A3_SIZES.items():
        if abs(w - expected_w) < 50 and abs(h - expected_h) < 50:
            return dpi

    estimated_dpi = int(w / 8.27)
    return max(150, min(600, estimated_dpi))


def detect_roi_type(image_path: str | Path) -> str:
    """Detect ROI type from image content: landscape→bottom_right, portrait→bottom, tall→right."""
    from pathlib import Path

    path_str = str(image_path).lower()

    if "fullpage" in path_str or "full_page" in path_str:
        return "full_page"

    path = Path(image_path) if not isinstance(image_path, Path) else image_path

    if not path.exists():
        return "bottom_right"

    try:
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    img = cv2.imread(str(f))
                    if img is not None:
                        h, w = img.shape[:2]
                        if h > w:
                            return "right" if h / w > 1.3 else "bottom"
                        return "bottom_right"
                    break
        else:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                if h > w:
                    return "right" if h / w > 1.3 else "bottom"
    except Exception:
        pass

    return "bottom_right"


def normalize_dpi(dpi_value: str | int | None) -> int | None:
    """Normalize DPI value: return int or None."""
    if dpi_value is None:
        return None
    if isinstance(dpi_value, int):
        return dpi_value
    if isinstance(dpi_value, str):
        if dpi_value.lower() == "auto":
            return None
        if dpi_value.isdigit():
            return int(dpi_value)
    return None


def calculate_roi_for_dpi(
    image_shape, dpi, roi_position: str = "bottom_right"
) -> tuple[int, int, int, int]:
    """Calculate ROI bbox for given DPI based on expected stamp size."""
    h, w = image_shape[:2]

    stamp_w = int(297 / 25.4 * dpi * 1.3)  # Form 5 width = 297mm
    stamp_h = int(115 / 25.4 * dpi * 1.3)  # Form 4 height = 115mm

    stamp_w = min(stamp_w, w - 20)
    stamp_h = min(stamp_h, h - 20)

    positions = {
        "bottom_right": (w - stamp_w, h - stamp_h, stamp_w, stamp_h),
        "bottom_left": (0, h - stamp_h, stamp_w, stamp_h),
        "top_right": (w - stamp_w, 0, stamp_w, stamp_h),
        "top_left": (0, 0, stamp_w, stamp_h),
        "bottom": (0, h - stamp_h, w, stamp_h),
        "top": (0, 0, w, stamp_h),
        "left": (0, 0, stamp_w, h),
        "right": (w - stamp_w, 0, stamp_w, h),
    }

    return positions.get(roi_position, positions["bottom_right"])


@dataclass
class PreprocessedImage:
    image: np.ndarray
    roi_image: np.ndarray
    roi_bbox: tuple[int, int, int, int]
    original_path: Path | None
    skew_angle: float
    flip_angle: int
    roi_position: str
    dpi: int | None = None
    dpi_roi: int | None = None
    filter_by_size: bool = True


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)

    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def denoise_image(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before thresholding."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def flip_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def detect_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        DESKEW_THRESHOLD,
        minLineLength=DESKEW_MIN_LINE_LENGTH,
        maxLineGap=DESKEW_MAX_LINE_GAP,
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 45 or abs(angle) > 135:
                angles.append(angle)

    if not angles:
        return 0.0

    hist, bins = np.histogram(angles, bins=180)
    peak_idx = np.argmax(hist)
    peak_angle = (bins[peak_idx] + bins[peak_idx + 1]) / 2

    return peak_angle


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    angle = detect_skew_angle(image)
    if abs(angle) < 0.5:
        return image, 0.0

    deskewed = rotate_image(image, angle)
    return deskewed, angle


def extract_roi(
    image: np.ndarray,
    roi_position: str = "bottom_right",
    adaptive: bool = False,
    dpi: int | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]

    if adaptive and dpi:
        x, y, roi_w, roi_h = calculate_roi_for_dpi(image.shape, dpi, roi_position)
    elif roi_position == "full_page":
        x, y, roi_w, roi_h = 0, 0, w, h
    else:
        w_ratio = ROI_WIDTH_RATIO
        h_ratio = ROI_HEIGHT_RATIO

        half_w = int(w * w_ratio)
        half_h = int(h * h_ratio)

        if roi_position == "bottom_right":
            x, y, roi_w, roi_h = w - half_w, h - half_h, half_w, half_h
        elif roi_position == "bottom_left":
            x, y, roi_w, roi_h = 0, h - half_h, half_w, half_h
        elif roi_position == "top_right":
            x, y, roi_w, roi_h = w - half_w, 0, half_w, half_h
        elif roi_position == "top_left":
            x, y, roi_w, roi_h = 0, 0, half_w, half_h
        elif roi_position == "bottom":
            x, y, roi_w, roi_h = 0, h - half_h, w, half_h
        elif roi_position == "top":
            x, y, roi_w, roi_h = 0, 0, w, half_h
        elif roi_position == "left":
            x, y, roi_w, roi_h = 0, 0, half_w, h
        elif roi_position == "right":
            x, y, roi_w, roi_h = w - half_w, 0, half_w, h
        else:
            x, y, roi_w, roi_h = w - half_w, h - half_h, half_w, half_h

    roi = image[y : y + roi_h, x : x + roi_w]

    return roi, (x, y, roi_w, roi_h)


def load_images(
    input_path: Path,
    flip_angles: list[int] | None = None,
    roi_position: str = "bottom_right",
    dpi_roi: int | None = None,
    filter_by_size: bool = True,
    debug: bool = False,
) -> list[PreprocessedImage]:
    input_path = Path(input_path)
    flip_angles = flip_angles or [0]

    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(f"Указанный путь не существует: {input_path}")

    image_files = []
    if input_path.is_file():
        if input_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            raise ValueError(f"Неподдерживаемый тип файла: {input_path.suffix}")
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [
            f
            for f in input_path.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]

    if not image_files:
        print(f"В '{input_path}' не найдено изображений для обработки.")
        return []

    print(f"Найдено файлов для обработки: {len(image_files)}")

    results: list[PreprocessedImage] = []

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(
                f"  Warning: Не удалось загрузить или поврежден файл: {img_path.name}. Пропускается."
            )
            continue

        if dpi_roi is not None:
            dpi = dpi_roi if isinstance(dpi_roi, int) else detect_dpi(image)
            use_dpi_roi = True
        else:
            dpi = None
            use_dpi_roi = False

        for flip_angle in flip_angles:
            flipped = flip_image(image, flip_angle) if flip_angle != 0 else image.copy()
            deskewed, skew_angle = deskew_image(flipped)
            roi_image, roi_bbox = extract_roi(
                deskewed,
                roi_position,
                adaptive=use_dpi_roi,
                dpi=dpi,
            )

            result = PreprocessedImage(
                image=deskewed,
                roi_image=roi_image,
                roi_bbox=roi_bbox,
                original_path=img_path,
                skew_angle=skew_angle,
                flip_angle=flip_angle,
                roi_position=roi_position,
                dpi=dpi,
                dpi_roi=dpi,
                filter_by_size=filter_by_size,
            )
            results.append(result)

            if debug:
                DEBUG_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
                DEBUG_PREPROCESSING_ROI_DIR.mkdir(parents=True, exist_ok=True)
                suffix = f"_flip{flip_angle}" if flip_angle != 0 else ""
                name = img_path.stem

                cv2.imwrite(
                    str(DEBUG_PREPROCESSING_DIR / f"{name}{suffix}_preprocessed.png"),
                    deskewed,
                )
                cv2.imwrite(
                    str(DEBUG_PREPROCESSING_ROI_DIR / f"{name}{suffix}_roi.png"),
                    roi_image,
                )

    print(f"Предобработано изображений: {len(results)}")
    for r in results:
        flip_info = f", flip={r.flip_angle}" if r.flip_angle != 0 else ""
        dpi_info = f", dpi={r.dpi}" if r.dpi else ""
        path_name = r.original_path.name if r.original_path else "unknown"
        print(f"  {path_name}: skew={r.skew_angle:.2f}{flip_info}{dpi_info}")

    return results
