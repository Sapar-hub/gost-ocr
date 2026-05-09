"""
Функции для расчёта метрик.

IoU (Intersection over Union) - основная метрика локализации.
"""

from dataclasses import dataclass
from typing import List, Optional


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