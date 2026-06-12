"""
Функции для расчёта метрик.

IoU (Intersection over Union) - основная метрика локализации.
Precision / Recall / F1 — качество детекции при пороге IoU ≥ 0.5.
Detection rate — доля изображений с найденным штампом.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectionResult:
    """Результат детекции для одного изображения."""
    image_name: str
    pred_bbox: Optional[List[float]]  # [x1, y1, x2, y2] в пикселях
    gt_bbox: Optional[List[float]]    # [x1, y1, x2, y2] в пикселях
    iou: float = 0.0
    has_prediction: bool = False
    has_ground_truth: bool = False
    method: str = ""


@dataclass
class AggregatedMetrics:
    """Агрегированные метрики для метода."""
    method: str
    mean_iou: float
    median_iou: float
    max_iou: float
    success_count: int  # IoU > threshold
    success_rate: float
    fail_count: int     # IoU == 0
    total: int


def box_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """
    Расчёт IoU между двумя bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] в пикселях
        box2: [x1, y1, x2, y2] в пикселях

    Returns:
        IoU в диапазоне [0, 1]
    """
    if box1 is None or box2 is None:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def yolo_label_to_bbox(label: str, img_w: int, img_h: int) -> Optional[List[float]]:
    """
    Конвертация YOLO формата (class xc yc w h) в xyxy.

    Args:
        label: строка из label файла, например "0 0.5 0.8 0.2 0.1"
        img_w: ширина изображения в пикселях
        img_h: высота изображения в пикселях

    Returns:
        [x1, y1, x2, y2] в пикселях
    """
    parts = label.strip().split()
    if len(parts) < 5:
        return None

    try:
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None

    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h

    return [x1, y1, x2, y2]


def aggregate_metrics(
    results: List[DetectionResult],
    threshold: float = 0.5
) -> AggregatedMetrics:
    """
    Агрегация метрик для списка результатов.
    """
    if not results:
        return AggregatedMetrics(
            method="",
            mean_iou=0.0,
            median_iou=0.0,
            max_iou=0.0,
            success_count=0,
            success_rate=0.0,
            fail_count=0,
            total=0
        )

    ious = [r.iou for r in results]
    success_count = sum(1 for iou in ious if iou > threshold)
    fail_count = sum(1 for iou in ious if iou == 0.0)

    import statistics
    mean_iou = statistics.mean(ious)
    median_iou = statistics.median(ious)
    max_iou = max(ious)

    method = results[0].method if results else ""

    return AggregatedMetrics(
        method=method,
        mean_iou=mean_iou,
        median_iou=median_iou,
        max_iou=max_iou,
        success_count=success_count,
        success_rate=success_count / len(results),
        fail_count=fail_count,
        total=len(results)
    )


def compare_methods(
    yolo_results: List[DetectionResult],
    opencv_results: List[DetectionResult]
) -> dict:
    """
    Сравнение двух методов по результатам детекции.

    Returns:
        {
            "yolo_wins": int,
            "opencv_wins": int,
            "ties": int
        }
    """
    yolo_wins = 0
    opencv_wins = 0
    ties = 0

    # Результаты должны быть в одинаковом порядке
    for yolo_r, opencv_r in zip(yolo_results, opencv_results):
        if yolo_r.iou > opencv_r.iou:
            yolo_wins += 1
        elif opencv_r.iou > yolo_r.iou:
            opencv_wins += 1
        else:
            ties += 1

    return {
        "yolo_wins": yolo_wins,
        "opencv_wins": opencv_wins,
        "ties": ties
    }


def compute_metrics(
    results: List[DetectionResult],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Полный расчёт метрик детекции.

    Возвращает словарь с метриками:
        n_images, n_found, detection_rate,
        iou_mean/std/median/max/min,
        iou_at_threshold, precision, recall, f1, tp, fp, fn
    """
    total = len(results)
    if total == 0:
        return {}

    found = [r for r in results if r.has_prediction]
    n_found = len(found)
    n_with_gt = len([r for r in results if r.has_ground_truth])

    ious = [r.iou for r in results]
    ious_found = [r.iou for r in found]

    tp = sum(1 for r in results if r.has_prediction and r.iou >= iou_threshold)
    fp = n_found - tp
    fn = n_with_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_images": total,
        "n_found": n_found,
        "detection_rate": n_found / total,
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "iou_median": float(np.median(ious)),
        "iou_max": float(np.max(ious)),
        "iou_min": float(np.min(ious)),
        "iou_found_mean": float(np.mean(ious_found)) if ious_found else float("nan"),
        "iou_at_threshold": sum(1 for i in ious if i >= iou_threshold) / total,
        "success_count": sum(1 for i in ious if i > iou_threshold),
        "fail_count": sum(1 for i in ious if i == 0.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_matching_bbox(
    pred_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит предсказанный bbox с наибольшим IoU относительно GT.
    Полезно когда модель возвращает несколько bbox, а GT один.
    """
    if not pred_bboxes:
        return None
    if not gt_bboxes:
        return pred_bboxes[0]

    best_iou_val = 0.0
    best_bbox = None
    for pb in pred_bboxes:
        for gt_bbox in gt_bboxes:
            iou_val = box_iou(list(pb), list(gt_bbox))
            if iou_val > best_iou_val:
                best_iou_val = iou_val
                best_bbox = pb
    return best_bbox


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Вывод метрик в терминал."""
    print(f"{prefix}Images:         {metrics.get('n_images', 'N/A')}")
    print(f"{prefix}Detected:        {metrics.get('n_found', 'N/A')} ({metrics.get('detection_rate', 0)*100:.1f}%)")
    print(f"{prefix}IoU mean:        {metrics.get('iou_mean', 0):.3f}")
    print(f"{prefix}IoU std:         {metrics.get('iou_std', 0):.3f}")
    print(f"{prefix}IoU median:      {metrics.get('iou_median', 0):.3f}")
    print(f"{prefix}IoU >= 0.5:      {metrics.get('iou_at_threshold', 0)*100:.1f}%")
    print(f"{prefix}Precision:       {metrics.get('precision', 0):.3f}")
    print(f"{prefix}Recall:          {metrics.get('recall', 0):.3f}")
    print(f"{prefix}F1:              {metrics.get('f1', 0):.3f}")