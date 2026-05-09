"""
Визуализация результатов сравнения методов локализации.

Создаёт side-by-side изображения с bbox от YOLO, OpenCV и ground truth.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from src.gost_ocr.benchmark.metrics import DetectionResult


def draw_bbox(
    image: np.ndarray,
    bbox: Optional[List[float]],
    color: tuple,
    thickness: int = 3,
    label: str = ""
) -> np.ndarray:
    """Рисует bbox на изображении."""
    if bbox is None:
        return image

    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Ограничиваем bbox размерами изображения
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    return image


def create_comparison_image(
    image_path: Path,
    yolo_result: DetectionResult,
    opencv_result: DetectionResult,
    gt_bbox: Optional[List[float]]
) -> Optional[np.ndarray]:
    """Создаёт side-by-side сравнение для одного изображения."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
    except Exception:
        return None

    h, w = img.shape[:2]
    # Ограничиваем max ширину для отображения
    max_width = 800
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
        # Масштабируем bbox
        scale_bbox = lambda b: [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] if b else None
        yolo_bbox = scale_bbox(yolo_result.pred_bbox)
        opencv_bbox = scale_bbox(opencv_result.pred_bbox)
        gt_bbox_scaled = scale_bbox(gt_bbox)
    else:
        yolo_bbox = yolo_result.pred_bbox
        opencv_bbox = opencv_result.pred_bbox
        gt_bbox_scaled = gt_bbox

    # Цвета
    color_gt = (0, 255, 0)      # зелёный - ground truth
    color_yolo = (255, 0, 0)    # синий - YOLO
    color_opencv = (0, 0, 255)  # красный - OpenCV

    # Создаём 4 панели: GT, YOLO, OpenCV, All
    panel_w = img.shape[1]
    panel_h = img.shape[0]

    # Панель 1: Ground Truth
    img_gt = img.copy()
    if gt_bbox_scaled:
        draw_bbox(img_gt, gt_bbox_scaled, color_gt, label="GT")

    # Панель 2: YOLO
    img_yolo = img.copy()
    if yolo_bbox:
        draw_bbox(img_yolo, yolo_bbox, color_yolo, label=f"YOLO IoU={yolo_result.iou:.2f}")

    # Панель 3: OpenCV
    img_opencv = img.copy()
    if opencv_bbox:
        draw_bbox(img_opencv, opencv_bbox, color_opencv, label=f"OpenCV IoU={opencv_result.iou:.2f}")

    # Панель 4: Все вместе
    img_all = img.copy()
    if gt_bbox_scaled:
        draw_bbox(img_all, gt_bbox_scaled, color_gt, label="GT")
    if yolo_bbox:
        draw_bbox(img_all, yolo_bbox, color_yolo, label="YOLO")
    if opencv_bbox:
        draw_bbox(img_all, opencv_bbox, color_opencv, label="OpenCV")

    # Объединяем в 2x2 сетку
    top = np.hstack([img_gt, img_yolo])
    bottom = np.hstack([img_opencv, img_all])
    result = np.vstack([top, bottom])

    # Добавляем заголовки
    h_img, w_img = result.shape[:2]
    header = np.zeros((40, w_img, 3), dtype=np.uint8)
    labels = ["Ground Truth", "YOLO", "OpenCV", "All"]
    for i, label in enumerate(labels):
        x = (i % 2) * w_img // 2 + 20
        y = 30
        cv2.putText(header, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    result = np.vstack([header, result])

    return result


def generate_visualizations(
    results: List[DetectionResult],
    opencv_results: List[DetectionResult],
    gt_data: dict,
    output_dir: Path,
    max_images: int = 20
):
    """Генерирует визуализации для всех изображений."""
    from src.gost_ocr.benchmark.constants import IMAGES_DIR

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Сортируем по суммарному IoU и берём топ-N
    yolo_dict = {r.image_name: r for r in results}
    opencv_dict = {r.image_name: r for r in opencv_results}

    # Вычисляем суммарный IoU и сортируем
    scored = []
    for name in yolo_dict:
        y = yolo_dict.get(name)
        o = opencv_dict.get(name)
        score = (y.iou if y else 0) + (o.iou if o else 0)
        scored.append((score, name))

    scored.sort(reverse=True)
    top_names = [name for _, name in scored[:max_images]]

    print(f"\n=== Генерация визуализаций (топ {len(top_names)}) ===")

    count = 0
    skipped = 0
    for image_name in top_names:
        yolo_result = yolo_dict.get(image_name)
        opencv_result = opencv_dict.get(image_name)

        if yolo_result is None:
            skipped += 1
            continue

        image_name = yolo_result.image_name
        opencv_result = opencv_dict.get(image_name)

        # Находим путь к изображению
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = IMAGES_DIR / f"{image_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        gt_bbox = gt_data.get(image_name)

        # Create placeholder result if None
        from src.gost_ocr.benchmark.metrics import DetectionResult
        if opencv_result is None:
            opencv_result = DetectionResult(
                image_name=image_name,
                pred_bbox=None,
                gt_bbox=None,
                iou=0.0,
                has_prediction=False,
                has_ground_truth=False,
                method="opencv"
            )

        comparison = create_comparison_image(
            img_path, yolo_result, opencv_result, gt_bbox
        )

        if comparison is not None:
            output_path = vis_dir / f"{image_name}_comparison.jpg"
            cv2.imwrite(str(output_path), comparison)
            count += 1

    print(f"Создано визуализаций: {count}")
    print(f"Сохранено в: {vis_dir}")

    return vis_dir