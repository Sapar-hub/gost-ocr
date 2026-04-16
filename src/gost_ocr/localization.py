from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import (
    DEBUG_LOCALIZATION_DIR,
    FORM_3_5_ASPECT_RATIO,
    MIN_STAMP_WIDTH_PX,
    MAX_STAMP_WIDTH_PX,
    MIN_STAMP_HEIGHT_PX,
    MIN_STAMP_AREA_PX,
)
from .preprocessing import PreprocessedImage, denoise_image, enhance_contrast


@dataclass
class StampCandidate:
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    area: int
    is_gost_compliant: bool
    form_type: str
    confidence: float
    depth: int


@dataclass
class LocalizationResult:
    stamp: StampCandidate | None
    all_candidates: list[StampCandidate]
    preprocessed: PreprocessedImage


def get_depth(i: int, hierarchy: np.ndarray, memo: dict | None = None) -> int:
    if memo is None:
        memo = {}
    if i not in memo:
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            memo[i] = 0
        else:
            memo[i] = 1 + get_depth(parent_idx, hierarchy, memo)
    return memo[i]


def check_gost_stamp_ratio(
    contour: np.ndarray, tolerance: float = 0.15, filter_by_size: bool = False
) -> tuple[bool, float, float]:
    x, y, w, h = cv2.boundingRect(contour)

    if h == 0:
        return False, 0.0, 0.0

    area = w * h

    if filter_by_size:
        if w < MIN_STAMP_WIDTH_PX or w > MAX_STAMP_WIDTH_PX:
            return False, 0.0, 0.0

        if h < MIN_STAMP_HEIGHT_PX:
            return False, 0.0, 0.0

        if area < MIN_STAMP_AREA_PX:
            return False, 0.0, 0.0

    aspect_ratio = w / h
    expected_ratio = FORM_3_5_ASPECT_RATIO

    ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio

    is_valid = ratio_diff <= tolerance

    confidence_score = max(0.0, 1.0 - (ratio_diff / tolerance))

    return is_valid, aspect_ratio, confidence_score


def _find_with_edge_detection(
    roi: np.ndarray, filter_by_size: bool = False
) -> list[StampCandidate]:
    """Fallback: find stamp using edge detection and contour analysis."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    gray = denoise_image(gray, kernel_size=5)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]
    candidates = []
    memo = {}

    for i, contour in enumerate(contours):
        if get_depth(i, hierarchy, memo) == 0:
            first_child_idx = hierarchy[i][2]

            if first_child_idx != -1:
                min_x, min_y = float("inf"), float("inf")
                max_x, max_y = float("-inf"), float("-inf")
                child_count = 0

                current_child_idx = first_child_idx
                while current_child_idx != -1:
                    if get_depth(current_child_idx, hierarchy, memo) == 1:
                        child_contour = contours[current_child_idx]
                        cx, cy, cw, ch = cv2.boundingRect(child_contour)
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx + cw)
                        max_y = max(max_y, cy + ch)
                        child_count += 1

                    current_child_idx = hierarchy[current_child_idx][0]

                if child_count > 1:
                    union_w = max_x - min_x
                    union_h = max_y - min_y

                    if union_w > 100 and union_h > 30:
                        is_valid, aspect, confidence = check_gost_stamp_ratio(
                            np.array(
                                [
                                    [[min_x, min_y]],
                                    [[max_x, min_y]],
                                    [[max_x, max_y]],
                                    [[min_x, max_y]],
                                ]
                            ),
                            filter_by_size=filter_by_size,
                        )

                        candidates.append(
                            StampCandidate(
                                bbox=(min_x, min_y, union_w, union_h),
                                aspect_ratio=aspect,
                                area=int(union_w * union_h),
                                is_gost_compliant=is_valid,
                                form_type="EDGE_DETECT" if is_valid else "EDGE_BOX",
                                confidence=confidence,
                                depth=0,
                            )
                        )

    return candidates


def find_stamp_contours(
    roi: np.ndarray,
    draw_all: bool = False,
    use_dynamic_threshold: bool = True,
    dpi: int | None = None,
    filter_by_size: bool = False,
) -> list[StampCandidate]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    denoised = denoise_image(gray, kernel_size=3)

    block_size = max(11, (dpi // 30) * 2 + 1) if dpi else 11
    if block_size % 2 == 0:
        block_size += 1

    c_constant = 5
    if dpi and dpi > 300:
        c_constant = 7

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_constant,
    )

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    candidates = []
    if hierarchy is None or len(contours) == 0:
        return candidates

    hierarchy = hierarchy[0]
    memo: dict = {}

    # In draw_all mode, we show original contours
    if draw_all:
        for i, contour in enumerate(contours):
            depth = get_depth(i, hierarchy, memo)
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(
                StampCandidate(
                    bbox=(x, y, w, h),
                    aspect_ratio=w / h if h > 0 else 0,
                    area=int(cv2.contourArea(contour)),
                    is_gost_compliant=False,
                    form_type=f"DEPTH_{depth}",
                    confidence=0.0,
                    depth=depth,
                )
            )
        return candidates

    # Strict mode: find union boxes of children from top-level containers
    for i, contour in enumerate(contours):
        if get_depth(i, hierarchy, memo) == 0:
            first_child_idx = hierarchy[i][2]

            if first_child_idx != -1:
                min_x, min_y = float("inf"), float("inf")
                max_x, max_y = float("-inf"), float("-inf")
                child_count = 0

                current_child_idx = first_child_idx
                while current_child_idx != -1:
                    if get_depth(current_child_idx, hierarchy, memo) == 1:
                        child_contour = contours[current_child_idx]
                        cx, cy, cw, ch = cv2.boundingRect(child_contour)
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx + cw)
                        max_y = max(max_y, cy + ch)
                        child_count += 1

                    current_child_idx = hierarchy[current_child_idx][0]

                if child_count > 1:
                    union_w = max_x - min_x
                    union_h = max_y - min_y
                    union_contour = np.array(
                        [
                            [[min_x, min_y]],
                            [[max_x, min_y]],
                            [[max_x, max_y]],
                            [[min_x, max_y]],
                        ]
                    )

                    is_valid, aspect, confidence = check_gost_stamp_ratio(
                        union_contour, filter_by_size=filter_by_size
                    )

                    candidate = StampCandidate(
                        bbox=(min_x, min_y, union_w, union_h),
                        aspect_ratio=aspect,
                        area=int(union_w * union_h),
                        is_gost_compliant=is_valid,
                        form_type="FORM_3_5" if is_valid else "UNION_BOX",
                        confidence=confidence,
                        depth=0,
                    )
                    candidates.append(candidate)

    # Dynamic threshold mode: always run, collect sizes for histogram analysis
    if use_dynamic_threshold:
        dynamic_candidates = _find_with_dynamic_threshold(
            roi, contours, hierarchy, memo
        )
        candidates = _merge_candidates(candidates, dynamic_candidates)

    # Fallback: if no candidates found, try edge detection
    gost_compliant = [c for c in candidates if c.is_gost_compliant]
    if not gost_compliant:
        edge_candidates = _find_with_edge_detection(roi, filter_by_size)
        if edge_candidates:
            candidates = _merge_candidates(candidates, edge_candidates)

    # Sort candidates
    gost_compliant = [c for c in candidates if c.is_gost_compliant]
    non_compliant = [c for c in candidates if not c.is_gost_compliant]

    gost_compliant.sort(key=lambda c: c.area, reverse=True)
    non_compliant.sort(key=lambda c: (c.confidence, c.area), reverse=True)

    candidates = gost_compliant + non_compliant

    return candidates


def _find_with_dynamic_threshold(
    roi: np.ndarray,
    contours: list,
    hierarchy: np.ndarray,
    memo: dict,
) -> list[StampCandidate]:
    """Find stamp using dynamic threshold based on size histogram."""

    if hierarchy is None or len(contours) == 0:
        return []

    # Collect all union boxes from parent contours (depth 0 with children)
    widths = []
    heights = []
    areas = []

    for i, contour in enumerate(contours):
        if get_depth(i, hierarchy, memo) == 0:
            first_child_idx = hierarchy[i][2]

            if first_child_idx != -1:
                min_x, min_y = float("inf"), float("inf")
                max_x, max_y = float("-inf"), float("-inf")
                child_count = 0

                current_child_idx = first_child_idx
                while current_child_idx != -1:
                    if get_depth(current_child_idx, hierarchy, memo) == 1:
                        child_contour = contours[current_child_idx]
                        cx, cy, cw, ch = cv2.boundingRect(child_contour)
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx + cw)
                        max_y = max(max_y, cy + ch)
                        child_count += 1

                    current_child_idx = hierarchy[current_child_idx][0]

                if child_count > 1:
                    union_w = max_x - min_x
                    union_h = max_y - min_y

                    if union_w > 100 and union_h > 30:  # Basic filter
                        widths.append(union_w)
                        heights.append(union_h)
                        areas.append(union_w * union_h)

    if not widths:
        return []

    # Find the mode (most common width) using histogram binning
    width_bins = np.histogram(widths, bins=50)[0]
    peak_bin_idx = np.argmax(width_bins)

    # Get the width range for this bin
    _, bin_edges = np.histogram(widths, bins=50)
    min_width = bin_edges[peak_bin_idx]
    max_width = bin_edges[peak_bin_idx + 1]

    # Accept candidates within ±30% of the peak
    tolerance = 0.30
    min_accepted = min_width * (1 - tolerance)
    max_accepted = max_width * (1 + tolerance)

    dynamic_candidates = []

    for i, contour in enumerate(contours):
        if get_depth(i, hierarchy, memo) == 0:
            first_child_idx = hierarchy[i][2]

            if first_child_idx != -1:
                min_x, min_y = float("inf"), float("inf")
                max_x, max_y = float("-inf"), float("-inf")
                child_count = 0

                current_child_idx = first_child_idx
                while current_child_idx != -1:
                    if get_depth(current_child_idx, hierarchy, memo) == 1:
                        child_contour = contours[current_child_idx]
                        cx, cy, cw, ch = cv2.boundingRect(child_contour)
                        min_x = min(min_x, cx)
                        min_y = min(min_y, cy)
                        max_x = max(max_x, cx + cw)
                        max_y = max(max_y, cy + ch)
                        child_count += 1

                    current_child_idx = hierarchy[current_child_idx][0]

                if child_count > 1:
                    union_w = max_x - min_x
                    union_h = max_y - min_y

                    if (
                        min_accepted <= union_w <= max_accepted
                        and union_h > MIN_STAMP_HEIGHT_PX * 0.5
                    ):
                        aspect = union_w / union_h if union_h > 0 else 0
                        expected_ratio = FORM_3_5_ASPECT_RATIO
                        ratio_diff = abs(aspect - expected_ratio) / expected_ratio
                        confidence = max(0.0, 1.0 - (ratio_diff / 0.30))

                        dynamic_candidates.append(
                            StampCandidate(
                                bbox=(min_x, min_y, union_w, union_h),
                                aspect_ratio=aspect,
                                area=int(union_w * union_h),
                                is_gost_compliant=True,  # Passes dynamic check
                                form_type="DYNAMIC",
                                confidence=confidence,
                                depth=0,
                            )
                        )

    return dynamic_candidates


def _merge_candidates(
    strict: list[StampCandidate],
    dynamic: list[StampCandidate],
) -> list[StampCandidate]:
    """Merge strict and dynamic candidates, preferring strict ones."""

    if not strict and dynamic:
        return dynamic

    if strict and not dynamic:
        return strict

    if not strict and not dynamic:
        return []

    # Use strict candidates as primary, add dynamic ones that aren't duplicates
    merged = list(strict)
    strict_bboxes = set(c.bbox for c in strict)

    for dc in dynamic:
        if dc.bbox not in strict_bboxes:
            merged.append(dc)

    return merged


def localize_stamp(
    preprocessed: PreprocessedImage,
    draw_all: bool = False,
    debug: bool = False,
) -> LocalizationResult:
    roi = preprocessed.roi_image
    roi_x, roi_y, _, _ = preprocessed.roi_bbox
    dpi = preprocessed.dpi
    filter_by_size = preprocessed.filter_by_size

    candidates = find_stamp_contours(
        roi, draw_all, dpi=dpi, filter_by_size=filter_by_size
    )

    if debug:
        DEBUG_LOCALIZATION_DIR.mkdir(parents=True, exist_ok=True)

        debug_roi = roi.copy()

        depth_colors = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
        }

        for c in candidates:
            x, y, w, h = c.bbox
            color = depth_colors.get(c.depth, (128, 128, 128))
            if c.is_gost_compliant:
                color = (0, 255, 0)
            cv2.rectangle(debug_roi, (x, y), (x + w, y + h), color, 2)
            label = f"{c.form_type} AR:{c.aspect_ratio:.2f} Conf:{c.confidence:.2f}"
            cv2.putText(
                debug_roi,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        suffix = (
            f"_flip{preprocessed.flip_angle}" if preprocessed.flip_angle != 0 else ""
        )
        name = (
            preprocessed.original_path.stem if preprocessed.original_path else "image"
        )

        cv2.imwrite(
            str(DEBUG_LOCALIZATION_DIR / f"{name}{suffix}_localization.png"), debug_roi
        )

    stamp = None
    if candidates and not draw_all:
        best = candidates[0]
        x, y, w, h = best.bbox
        global_bbox = (x + roi_x, y + roi_y, w, h)
        stamp = StampCandidate(
            bbox=global_bbox,
            aspect_ratio=best.aspect_ratio,
            area=best.area,
            is_gost_compliant=best.is_gost_compliant,
            form_type=best.form_type,
            confidence=best.confidence,
            depth=best.depth,
        )

    return LocalizationResult(
        stamp=stamp,
        all_candidates=candidates,
        preprocessed=preprocessed,
    )


def localize_images(
    preprocessed_images: list[PreprocessedImage],
    draw_all: bool = False,
    debug: bool = False,
) -> list[LocalizationResult]:
    results = []

    for preprocessed in preprocessed_images:
        result = localize_stamp(preprocessed, draw_all, debug)
        results.append(result)

        if draw_all:
            print(f"  Найдено контуров: {len(result.all_candidates)}")
        elif result.stamp:
            s = result.stamp
            print(
                f"  Штамп найден: {s.form_type}, bbox={s.bbox}, confidence={s.confidence:.2f}, depth={s.depth}"
            )
        else:
            print(f"  Штамп не найден")

    return results
