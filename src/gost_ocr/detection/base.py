from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class DetectionResult:
    """Result from detector - contains stamp bounding box and metadata."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    method: str  # "yolo" or "opencv"
    raw_result: Protocol | None = None  # Raw detection result from model


class Detector(ABC):
    """Abstract base class for stamp detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult | None:
        """Detect stamp in image. Returns DetectionResult or None if not found."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass