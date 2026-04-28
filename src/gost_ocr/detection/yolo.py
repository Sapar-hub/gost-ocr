from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from ultralytics import YOLO

from ..config import YOLO_TRAINED_MODEL
from .base import Detector, DetectionResult


class YoloDetector(Detector):
    """YOLO-based stamp detector."""

    def __init__(self, model_path: Path | str | None = None, device: str = "cpu"):
        if model_path is None:
            model_path = YOLO_TRAINED_MODEL
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}. "
                "Run training first: yolo detect train ..."
            )
        
        self.model = YOLO(str(self.model_path))
        self.device = device

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        """Detect stamp using YOLO."""
        results = self.model(image, verbose=False, device=self.device)
        
        if not results or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = box
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        return DetectionResult(
            bbox=bbox,
            confidence=float(conf),
            method="yolo",
            raw_result=results[0]
        )

    def get_name(self) -> str:
        return f"YOLO({self.model_path.name})"