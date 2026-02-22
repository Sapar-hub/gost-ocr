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


@dataclass
class PreprocessedImage:
    image: np.ndarray
    roi_image: np.ndarray
    roi_bbox: tuple[int, int, int, int]
    original_path: Path | None
    skew_angle: float
    flip_angle: int
    roi_position: str


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


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
    image: np.ndarray, roi_position: str = "bottom_right"
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]

    # Ratios for width and height
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
    else:  # Default to bottom_right
        x, y, roi_w, roi_h = w - half_w, h - half_h, half_w, half_h

    roi = image[y : y + roi_h, x : x + roi_w]

    return roi, (x, y, roi_w, roi_h)


def load_images(
    input_path: Path,
    flip_angles: list[int] | None = None,
    roi_position: str = "bottom_right",
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
            print(f"  Warning: Не удалось загрузить или поврежден файл: {img_path.name}. Пропускается.")
            continue

        for flip_angle in flip_angles:
            flipped = flip_image(image, flip_angle) if flip_angle != 0 else image.copy()
            deskewed, skew_angle = deskew_image(flipped)
            roi_image, roi_bbox = extract_roi(deskewed, roi_position)

            result = PreprocessedImage(
                image=deskewed,
                roi_image=roi_image,
                roi_bbox=roi_bbox,
                original_path=img_path,
                skew_angle=skew_angle,
                flip_angle=flip_angle,
                roi_position=roi_position,
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
        path_name = r.original_path.name if r.original_path else "unknown"
        print(f"  {path_name}: skew={r.skew_angle:.2f}{flip_info}")

    return results
