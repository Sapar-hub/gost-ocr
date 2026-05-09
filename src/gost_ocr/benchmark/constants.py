"""
Константы для evaluation модуля.

ROI маппинг для тестовой выборки:
- landscape -> bottom_right
- portrait -> bottom
"""

from pathlib import Path
from typing import Dict

# Пути
DATASET_DIR = Path("src/gost_ocr/datasets")
IMAGES_DIR = DATASET_DIR / "images" / "test"
LABELS_DIR = DATASET_DIR / "labels" / "test"
MODEL_PATH = Path("src/gost_ocr/models/yolo/best.pt")

# ROI маппинг для тестовой выборки (из ТЗ)
# test_01-15, 17, 19, 22, 29-49 -> bottom_right (landscape)
# test_16, 18, 21, 25-28 -> bottom (portrait)
# test_20, 24 -> right (portrait, альтернативная ориентация)
DEFAULT_ROI_MAP: Dict[str, str] = {
    **{f"test_{i:02d}": "bottom_right" for i in range(1, 16)},
    **{f"test_{i:02d}": "bottom_right" for i in [17, 19, 22]},
    **{f"test_{i:02d}": "bottom_right" for i in range(29, 50)},
    **{f"test_{i:02d}": "bottom" for i in [16, 18, 21, 25, 26, 27, 28]},
    **{f"test_{i:02d}": "right" for i in [20, 24]},
}

# Пороги
IoU_SUCCESS_THRESHOLD = 0.5  # IoU > 0.5 считается успешной локализацией