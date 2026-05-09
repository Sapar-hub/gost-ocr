#!/usr/bin/env python
"""
Основной скрипт для сравнения методов локализации штампов.

Сравнивает YOLO и OpenCV на тестовой выборке и выводит метрики в терминал.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from src.gost_ocr.benchmark.constants import (
    DEFAULT_ROI_MAP,
    IMAGES_DIR,
    LABELS_DIR,
    IoU_SUCCESS_THRESHOLD,
    MODEL_PATH,
)
from src.gost_ocr.benchmark.metrics import (
    DetectionResult,
    aggregate_metrics,
    box_iou,
    compare_methods,
    yolo_label_to_bbox,
)

# Импорт детекторов
from ultralytics import YOLO
from src.gost_ocr.detection.opencv import OpenCvDetector


def load_ground_truth() -> dict:
    """
    Загрузка ground truth labels.

    Returns:
        dict: {image_name_without_ext: [x1, y1, x2, y2] в пикселях}
    """
    gt_data = {}

    if not LABELS_DIR.exists():
        print(f"Warning: Labels directory not found: {LABELS_DIR}")
        return gt_data

    for label_file in LABELS_DIR.glob("*.txt"):
        # Извлекаем имя изображения из filename (убираем хеш префикс)
        # Например: "0c5a783a-test_05.txt" -> "test_05"
        filename = label_file.stem
        if "-" in filename:
            image_name = filename.split("-", 1)[1]
        else:
            image_name = filename

        with open(label_file) as f:
            label_content = f.read().strip()

        # Находим соответствующее изображение для получения размеров
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = IMAGES_DIR / f"{image_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
            bbox = yolo_label_to_bbox(label_content, w, h)
            if bbox:
                gt_data[image_name] = bbox
        except Exception as e:
            print(f"Warning: Failed to load GT for {image_name}: {e}")
            continue

    return gt_data


def get_image_path(image_name: str) -> Optional[Path]:
    """Находит путь к изображению по имени."""
    for ext in [".png", ".jpg", ".jpeg"]:
        path = IMAGES_DIR / f"{image_name}{ext}"
        if path.exists():
            return path
    return None


def run_yolo_detection(gt_data: dict) -> List[DetectionResult]:
    """
    Запускает YOLO детекцию на всех тестовых изображениях.
    """
    print("\n=== Запуск YOLO детекции ===")

    if not MODEL_PATH.exists():
        print(f"Error: Model not found: {MODEL_PATH}")
        return []

    model = YOLO(str(MODEL_PATH))
    results = []

    # Получаем список всех изображений
    image_files = sorted(IMAGES_DIR.glob("*"))
    image_names = [f.stem for f in image_files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    print(f"Обработка {len(image_names)} изображений...")

    for i, image_name in enumerate(image_names, 1):
        img_path = get_image_path(image_name)
        if img_path is None:
            continue

        # Предсказание YOLO
        try:
            yolo_result = model(str(img_path), verbose=False, device="cpu")
            pred_boxes = yolo_result[0].boxes

            if len(pred_boxes) > 0:
                # Берём bbox с наибольшей уверенностью
                confidences = pred_boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()
                xyxy = pred_boxes.xyxy[best_idx].cpu().numpy().tolist()
                pred_bbox = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
            else:
                pred_bbox = None
        except Exception as e:
            print(f"Warning: YOLO failed for {image_name}: {e}")
            pred_bbox = None

        gt_bbox = gt_data.get(image_name)

        result = DetectionResult(
            image_name=image_name,
            pred_bbox=pred_bbox,
            gt_bbox=gt_bbox,
            iou=box_iou(pred_bbox, gt_bbox),
            has_prediction=pred_bbox is not None,
            has_ground_truth=gt_bbox is not None,
            method="YOLO"
        )
        results.append(result)

        if i % 10 == 0:
            print(f"  Обработано {i}/{len(image_names)}")

    print(f"YOLO: {sum(1 for r in results if r.has_prediction)}/{len(results)} обнаружений")
    return results


def run_opencv_detection(gt_data: dict, roi_map: dict) -> List[DetectionResult]:
    """
    Запускает OpenCV детекцию на всех тестовых изображениях.
    """
    print("\n=== Запуск OpenCV детекции ===")

    # Импорт preprocessing функций
    from src.gost_ocr.preprocessing import load_images, PreprocessedImage
    from src.gost_ocr.localization import localize_stamp

    results = []

    # Группируем изображения по ROI
    roi_groups: dict[str, list[str]] = {}
    for image_name, roi in roi_map.items():
        if roi not in roi_groups:
            roi_groups[roi] = []
        roi_groups[roi].append(image_name)

    image_files = sorted(IMAGES_DIR.glob("*"))
    image_names = [f.stem for f in image_files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    print(f"Обработка {len(image_names)} изображений...")

    # Для каждого ROI запускаем детекцию
    for roi, names in roi_groups.items():
        # Фильтруем только нужные изображения
        # Создаём временную директорию с изображениями для этого ROI
        from pathlib import Path
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Копируем изображения во временную директорию
            for name in names:
                src = get_image_path(name)
                if src:
                    dst = Path(tmpdir) / src.name
                    shutil.copy(src, dst)

            # Загружаем с правильным ROI
            preprocessed_images = load_images(
                Path(tmpdir),
                flip_angles=[0],
                roi_position=roi,
                filter_by_size=False,
            )

            # Для каждого изображения запускаем локализацию
            for preproc in preprocessed_images:
                image_name = preproc.original_path.stem

                try:
                    loc_result = localize_stamp(preproc, draw_all=False, debug=False)
                    if loc_result.stamp and loc_result.stamp.bbox:
                        x1, y1, w, h = loc_result.stamp.bbox
                        pred_bbox = [x1, y1, x1 + w, y1 + h]
                    else:
                        pred_bbox = None
                except Exception as e:
                    pred_bbox = None

                gt_bbox = gt_data.get(image_name)

                result = DetectionResult(
                    image_name=image_name,
                    pred_bbox=pred_bbox,
                    gt_bbox=gt_bbox,
                    iou=box_iou(pred_bbox, gt_bbox),
                    has_prediction=pred_bbox is not None,
                    has_ground_truth=gt_bbox is not None,
                    method="OpenCV"
                )
                results.append(result)

    # Сортируем результаты по имени изображения для консистентности
    results.sort(key=lambda x: x.image_name)

    print(f"OpenCV: {sum(1 for r in results if r.has_prediction)}/{len(results)} обнаружений")
    return results


def print_results(
    yolo_results: List[DetectionResult],
    opencv_results: List[DetectionResult],
    comparison: dict
):
    """Вывод результатов в терминал."""
    yolo_metrics = aggregate_metrics(yolo_results, IoU_SUCCESS_THRESHOLD)
    opencv_metrics = aggregate_metrics(opencv_results, IoU_SUCCESS_THRESHOLD)

    print("\n" + "=" * 60)
    print("           СРАВНЕНИЕ МЕТОДОВ ЛОКАЛИЗАЦИИ ШТАМПОВ")
    print("=" * 60)
    print(f"\nТестовых изображений: {len(yolo_results)}")
    print(f"Порог успешной локализации: IoU > {IoU_SUCCESS_THRESHOLD}\n")

    # Таблица
    header = f"| {'Метод':<10} | {'Mean IoU':>9} | {'Median IoU':>11} | {'IoU>0.5':>7} | {'IoU=0':>6} |"
    separator = "-" * len(header)

    print(header)
    print(separator)
    print(f"| {'YOLO':<10} | {yolo_metrics.mean_iou:>9.3f} | {yolo_metrics.median_iou:>11.3f} | {yolo_metrics.success_count:>7} | {yolo_metrics.fail_count:>6} |")
    print(f"| {'OpenCV':<10} | {opencv_metrics.mean_iou:>9.3f} | {opencv_metrics.median_iou:>11.3f} | {opencv_metrics.success_count:>7} | {opencv_metrics.fail_count:>6} |")
    print(separator)

    print(f"\nПобедитель: ", end="")
    if comparison["opencv_wins"] > comparison["yolo_wins"]:
        print(f"OpenCV ({comparison['opencv_wins']}/{len(yolo_results)})")
    elif comparison["yolo_wins"] > comparison["opencv_wins"]:
        print(f"YOLO ({comparison['yolo_wins']}/{len(yolo_results)})")
    else:
        print(f"Ничья ({comparison['ties']}/{len(yolo_results)})")

    # Детали по каждому изображению (топ и фейлы)
    print("\n--- Лучшие результаты YOLO (TOP 5) ---")
    sorted_yolo = sorted(yolo_results, key=lambda x: x.iou, reverse=True)
    for r in sorted_yolo[:5]:
        print(f"  {r.image_name}: IoU={r.iou:.3f}")

    print("\n--- Лучшие результаты OpenCV (TOP 5) ---")
    sorted_opencv = sorted(opencv_results, key=lambda x: x.iou, reverse=True)
    for r in sorted_opencv[:5]:
        print(f"  {r.image_name}: IoU={r.iou:.3f}")

    print("\n" + "=" * 60)


def main():
    """Основная функция."""
    print("=" * 60)
    print("        СРАВНЕНИЕ YOLO vs OpenCV")
    print("        Локализация штампов технической документации")
    print("=" * 60)
    print(f"\nВремя запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Загрузка ground truth
    print("\nЗагрузка ground truth labels...")
    gt_data = load_ground_truth()
    print(f"Загружено GT: {len(gt_data)} изображений")

    if not gt_data:
        print("Error: No ground truth labels found!")
        return

    # Запуск детекции
    yolo_results = run_yolo_detection(gt_data)
    opencv_results = run_opencv_detection(gt_data, DEFAULT_ROI_MAP)

    if not yolo_results or not opencv_results:
        print("Error: No detection results!")
        return

    # Сравнение
    comparison = compare_methods(yolo_results, opencv_results)

    # Вывод
    print_results(yolo_results, opencv_results, comparison)

    # Сохранение детальных результатов в JSON (для отладки/анализа)
    output_dir = Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Визуализация (опционально)
    from src.gost_ocr.benchmark.visualize import generate_visualizations
    vis_dir = generate_visualizations(yolo_results, opencv_results, gt_data, output_dir)
    print(f"\nВизуализации: {vis_dir}")

    # Сохраняем сырые результаты
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(yolo_results),
        "yolo": [
            {
                "image": r.image_name,
                "iou": r.iou,
                "has_prediction": r.has_prediction,
                "has_gt": r.has_ground_truth
            }
            for r in yolo_results
        ],
        "opencv": [
            {
                "image": r.image_name,
                "iou": r.iou,
                "has_prediction": r.has_prediction,
                "has_gt": r.has_ground_truth
            }
            for r in opencv_results
        ],
        "comparison": comparison
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nДетальные результаты сохранены в: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()