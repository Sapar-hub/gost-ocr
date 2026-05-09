from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Metrics for a single image evaluation."""

    image_name: str
    iou: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    localization_success: bool = False
    text_success: bool = False
    notes: str = ""


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for a batch of images."""

    total_images: int = 0
    localization_success_count: int = 0
    text_success_count: int = 0

    mean_iou: float = 0.0
    median_iou: float = 0.0
    mean_cer: float = 0.0
    mean_wer: float = 0.0

    per_image: list[EvaluationMetrics] = field(default_factory=list)


def calculate_iou(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box as (x, y, w, h)
        box2: Second bounding box as (x, y, w, h)

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2

    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_cer(predicted: str, reference: str) -> float:
    """
    Calculate Character Error Rate.

    Args:
        predicted: Predicted text
        reference: Ground truth text

    Returns:
        CER score (0 = perfect, higher = more errors)
    """
    if not reference:
        return 1.0 if predicted else 0.0

    distance = _levenshtein_distance(predicted, reference)
    return distance / len(reference)


def calculate_wer(predicted: str, reference: str) -> float:
    """
    Calculate Word Error Rate.

    Args:
        predicted: Predicted text
        reference: Ground truth text

    Returns:
        WER score (0 = perfect, higher = more errors)
    """
    pred_words = predicted.split()
    ref_words = reference.split()

    if not ref_words:
        return 1.0 if pred_words else 0.0

    distance = _levenshtein_distance(pred_words, ref_words)
    return distance / len(ref_words)


def _levenshtein_distance(s1: str | list, s2: str | list) -> int:
    """
    Calculate Levenshtein distance between two sequences.
    Works with both strings (character-level) and lists (word-level).
    """
    if isinstance(s1, str):
        s1 = list(s1)
    if isinstance(s2, str):
        s2 = list(s2)

    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def load_ground_truth(gt_path: Path) -> dict | None:
    """Load ground truth JSON file."""
    if not gt_path.exists():
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ocr_result(result_path: Path) -> dict | None:
    """Load OCR result JSON file."""
    if not result_path.exists():
        return None

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(
    ground_truth: dict,
    ocr_result: dict,
    iou_threshold: float = 0.5,
    cer_threshold: float = 0.1,
    wer_threshold: float = 0.2,
) -> EvaluationMetrics:
    """
    Evaluate a single image against ground truth.

    Args:
        ground_truth: Ground truth data dict
        ocr_result: OCR output data dict
        iou_threshold: Minimum IoU for successful localization
        cer_threshold: Maximum CER for successful text recognition
        wer_threshold: Maximum WER for successful text recognition

    Returns:
        EvaluationMetrics for this image
    """
    image_name = Path(ground_truth.get("image_path", "")).name

    gt_bbox = tuple(ground_truth.get("stamp_bbox", [0, 0, 0, 0]))
    pred_bbox = tuple(ocr_result.get("stamp_bbox", [0, 0, 0, 0]))

    iou = calculate_iou(gt_bbox, pred_bbox)
    localization_success = iou >= iou_threshold

    gt_text = ground_truth.get("text", "")
    pred_text = ocr_result.get("full_text", "")

    cer = calculate_cer(pred_text, gt_text)
    wer = calculate_wer(pred_text, gt_text)
    text_success = cer <= cer_threshold and wer <= wer_threshold

    return EvaluationMetrics(
        image_name=image_name,
        iou=iou,
        cer=cer,
        wer=wer,
        localization_success=localization_success,
        text_success=text_success,
    )


def evaluate_batch(
    ground_truth_dir: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    cer_threshold: float = 0.1,
    wer_threshold: float = 0.2,
) -> EvaluationResult:
    """
    Evaluate a batch of images against ground truth.

    Args:
        ground_truth_dir: Directory containing ground truth JSON files
        output_dir: Directory containing OCR output JSON files
        iou_threshold: Minimum IoU for successful localization
        cer_threshold: Maximum CER for successful text recognition
        wer_threshold: Maximum WER for successful text recognition

    Returns:
        Aggregated EvaluationResult
    """
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)

    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    gt_files = list(ground_truth_dir.glob("*.json"))

    results: list[EvaluationMetrics] = []

    for gt_file in gt_files:
        gt_data = load_ground_truth(gt_file)
        if gt_data is None:
            continue

        image_name = Path(gt_data.get("image_path", "")).name

        result_file = output_dir / f"{Path(image_name).stem}_output.json"
        if not result_file.exists():
            results.append(
                EvaluationMetrics(
                    image_name=image_name,
                    notes="No output file found",
                )
            )
            continue

        ocr_data = load_ocr_result(result_file)
        if ocr_data is None:
            results.append(
                EvaluationMetrics(
                    image_name=image_name,
                    notes="Failed to load output file",
                )
            )
            continue

        metrics = evaluate_single(
            gt_data, ocr_data, iou_threshold, cer_threshold, wer_threshold
        )
        results.append(metrics)

    valid_results = [r for r in results if r.notes == ""]
    valid_ious = [r.iou for r in valid_results]
    valid_cers = [r.cer for r in valid_results]
    valid_wers = [r.wer for r in valid_results]

    loc_success = sum(1 for r in valid_results if r.localization_success)
    txt_success = sum(1 for r in valid_results if r.text_success)

    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    median_iou = sorted(valid_ious)[len(valid_ious) // 2] if valid_ious else 0.0
    mean_cer = sum(valid_cers) / len(valid_cers) if valid_cers else 0.0
    mean_wer = sum(valid_wers) / len(valid_wers) if valid_wers else 0.0

    return EvaluationResult(
        total_images=len(results),
        localization_success_count=loc_success,
        text_success_count=txt_success,
        mean_iou=mean_iou,
        median_iou=median_iou,
        mean_cer=mean_cer,
        mean_wer=mean_wer,
        per_image=results,
    )


def print_evaluation_report(result: EvaluationResult) -> None:
    """Print evaluation report to console."""
    print("\n" + "=" * 50)
    print("=== Evaluation Results ===")
    print("=" * 50)

    print(f"\nImages processed: {result.total_images}")
    print(
        f"Localization success: {result.localization_success_count}/{result.total_images}"
    )
    print(
        f"Text recognition success: {result.text_success_count}/{result.total_images}"
    )

    print(f"\n--- Aggregate Metrics ---")
    print(
        f"Localization IoU:  {result.mean_iou:.2f} (mean), {result.median_iou:.2f} (median)"
    )
    print(f"Text CER:          {result.mean_cer:.2f} (mean)")
    print(f"Text WER:          {result.mean_wer:.2f} (mean)")

    print(f"\n--- Per-Image Details ---")
    for metric in result.per_image:
        status = "OK" if metric.localization_success and metric.text_success else "FAIL"
        if metric.notes:
            print(f"  {metric.image_name}: [{status}] {metric.notes}")
        else:
            print(
                f"  {metric.image_name}: [{status}] IoU={metric.iou:.2f}, CER={metric.cer:.2f}, WER={metric.wer:.2f}"
            )

    print("=" * 50 + "\n")


def save_evaluation_report(result: EvaluationResult, output_path: Path) -> None:
    """Save evaluation report to JSON file."""
    import dataclasses

    report = {
        "total_images": result.total_images,
        "localization_success_count": result.localization_success_count,
        "text_success_count": result.text_success_count,
        "mean_iou": result.mean_iou,
        "median_iou": result.median_iou,
        "mean_cer": result.mean_cer,
        "mean_wer": result.mean_wer,
        "per_image": [
            {
                "image_name": m.image_name,
                "iou": m.iou,
                "cer": m.cer,
                "wer": m.wer,
                "localization_success": m.localization_success,
                "text_success": m.text_success,
                "notes": m.notes,
            }
            for m in result.per_image
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
