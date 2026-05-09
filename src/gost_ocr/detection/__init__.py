"""Stamp detection module - supports YOLO and OpenCV methods."""

from .base import Detector, DetectionResult
from .factory import get_detector, get_detector_info, DetectorType
from .opencv import OpenCvDetector
from .yolo import YoloDetector

__all__ = [
    "Detector",
    "DetectionResult",
    "DetectorType",
    "get_detector",
    "get_detector_info",
    "OpenCvDetector",
    "YoloDetector",
]