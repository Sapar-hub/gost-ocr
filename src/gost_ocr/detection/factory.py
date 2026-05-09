from pathlib import Path
from typing import Literal

from .base import Detector
from .opencv import OpenCvDetector
from .yolo import YoloDetector


DetectorType = Literal["auto", "yolo", "opencv"]


def get_detector(
    method: DetectorType = "auto",
    yolo_path: Path | str | None = None,
    opencv_filter_by_size: bool = False,
) -> Detector:
    """
    Factory function to get a detector instance.
    
    Args:
        method: Detection method - "auto" (YOLO with OpenCV fallback),
                "yolo" (YOLO only), or "opencv" (OpenCV only)
        yolo_path: Path to YOLO model weights (optional)
        opencv_filter_by_size: Enable size filtering for OpenCV detector
    
    Returns:
        Detector instance
    """
    if method == "opencv":
        return OpenCvDetector(filter_by_size=opencv_filter_by_size)

    if method == "yolo":
        return YoloDetector(model_path=yolo_path)

    # Auto mode: try YOLO first, fallback to OpenCV
    try:
        detector = YoloDetector(model_path=yolo_path)
        return detector
    except FileNotFoundError:
        # YOLO model not trained yet, use OpenCV
        return OpenCvDetector(filter_by_size=opencv_filter_by_size)


def get_detector_info() -> dict[str, bool]:
    """Return info about available detectors."""
    from ..config import YOLO_TRAINED_MODEL
    
    return {
        "yolo_available": YOLO_TRAINED_MODEL.exists(),
        "opencv_available": True,
    }