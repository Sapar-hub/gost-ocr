import numpy as np

from ..localization import localize_stamp
from ..preprocessing import PreprocessedImage
from .base import Detector, DetectionResult


class OpenCvDetector(Detector):
    """OpenCV-based stamp detector using contour analysis."""

    def __init__(self, filter_by_size: bool = False):
        self.filter_by_size = filter_by_size

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        """Detect stamp using OpenCV contour analysis."""
        preprocessed = PreprocessedImage(
            image=image,
            roi_image=image,
            roi_bbox=(0, 0, image.shape[1], image.shape[0]),
            original_path=None,
            skew_angle=0.0,
            flip_angle=0,
            roi_position="full_page",
            filter_by_size=self.filter_by_size,
        )

        result = localize_stamp(preprocessed, draw_all=False, debug=False)

        if result.stamp is None:
            return None

        return DetectionResult(
            bbox=result.stamp.bbox,
            confidence=result.stamp.confidence,
            method="opencv",
            raw_result=result
        )

    def get_name(self) -> str:
        return "OpenCV"