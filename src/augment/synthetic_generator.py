from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_URLS = [
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "https://github.com/stephenjenkins/arial-unicode-ms/raw/master/arial-unicode-ms.ttf",
]

_cached_font_path = None


def get_font_path() -> str:
    """Get font path with Cyrillic support (cached)."""
    global _cached_font_path
    if _cached_font_path is not None:
        return _cached_font_path

    font_dir = Path.home() / ".gost_ocr" / "fonts"
    font_path = font_dir / "DejaVuSans.ttf"

    if not font_path.exists():
        font_dir.mkdir(parents=True, exist_ok=True)

        for url in FONT_URLS:
            try:
                urllib.request.urlretrieve(url, font_path)
                _cached_font_path = str(font_path)
                return _cached_font_path
            except Exception:
                continue

    system_font_paths = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]

    for fp in system_font_paths:
        if Path(fp).exists():
            _cached_font_path = fp
            return _cached_font_path

    _cached_font_path = ""
    return _cached_font_path


def draw_russian_text(
    img_array: np.ndarray, text: str, position: tuple[int, int], font_size: int
) -> np.ndarray:
    """Draw Russian text on image using PIL."""
    font_path = get_font_path()

    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)

    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(0, 0, 0))
    return np.array(img_pil)


@dataclass
class GOSTForm:
    """GOST form specification."""

    name: str
    width_mm: float
    height_mm: float
    rows: int
    cols: int

    @property
    def aspect_ratio(self) -> float:
        return self.width_mm / self.height_mm


GOST_FORMS = {
    "FORM_3": GOSTForm("Форма 3", 185, 55, 5, 9),
    "FORM_4": GOSTForm("Форма 4", 185, 115, 8, 9),
    "FORM_5": GOSTForm("Форма 5", 297, 55, 5, 15),
}


def mm_to_pixels(mm: float, dpi: int) -> int:
    """Convert millimeters to pixels for given DPI."""
    return int(mm * dpi / 25.4)


def create_stamp_grid(
    form: GOSTForm,
    dpi: int,
    add_text: bool = True,
) -> np.ndarray:
    """Create a stamp grid image for a given GOST form."""
    w = mm_to_pixels(form.width_mm, dpi)
    h = mm_to_pixels(form.height_mm, dpi)

    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    cell_w = w // form.cols
    cell_h = h // form.rows

    line_color = (0, 0, 0)
    thickness = max(1, dpi // 100)

    for i in range(form.rows + 1):
        y = i * cell_h
        cv2.line(img, (0, y), (w, y), line_color, thickness)

    for j in range(form.cols + 1):
        x = j * cell_w
        cv2.line(img, (x, 0), (x, h), line_color, thickness)

    if add_text:
        font_size = max(28, dpi // 8)

        text_samples = ["ИЗМ", "Лист", "№ докум.", "Подпись", "Дата"]
        for i in range(min(form.rows, len(text_samples))):
            y = i * cell_h + cell_h // 2
            img = draw_russian_text(
                img, text_samples[i], (cell_w // 4, y - font_size // 2), font_size
            )

        for j in range(min(form.cols, 3)):
            x = j * cell_w + cell_w // 2 - 10
            img = draw_russian_text(
                img, str(j + 1), (x, cell_h // 2 - font_size // 2), font_size
            )

        for i in range(form.rows):
            for j in range(form.cols):
                if random.random() < 0.3:
                    x = j * cell_w + cell_w // 4
                    y = i * cell_h + cell_h // 2
                    fill_text = random.choice(["", "А", "Б", "В", "Г", "1", "2", "3"])
                    if fill_text:
                        img = draw_russian_text(
                            img, fill_text, (x, y - font_size // 2), font_size
                        )

    return img


def add_noise(image: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Add random noise to simulate scanned document."""
    if noise_level <= 0:
        return image

    h, w = image.shape[:2]
    noise = np.random.normal(0, 255 * noise_level, (h, w, 3))
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def add_blur(image: np.ndarray, blur_probability: float = 0.3) -> np.ndarray:
    """Add slight blur to simulate out-of-focus scanning."""
    if random.random() > blur_probability:
        return image

    kernel_size = random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_rotation(image: np.ndarray, max_angle: float = 2.0) -> np.ndarray:
    """Add slight rotation to simulate imperfect scanning."""
    if max_angle <= 0:
        return image

    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def add_gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


def add_speckle_noise(image: np.ndarray, intensity: float = 0.01) -> np.ndarray:
    """Add speckle noise (salt and pepper)."""
    if intensity <= 0:
        return image

    result = image.copy()
    h, w = image.shape[:2]

    num_salt = int(h * w * intensity / 2)
    num_pepper = int(h * w * intensity / 2)

    salt_coords = (
        [random.randint(0, h - 1) for _ in range(num_salt)],
        [random.randint(0, w - 1) for _ in range(num_salt)],
    )
    pepper_coords = (
        [random.randint(0, h - 1) for _ in range(num_pepper)],
        [random.randint(0, w - 1) for _ in range(num_pepper)],
    )

    result[salt_coords] = 255
    result[pepper_coords] = 0

    return result


def add_ink_bleeding(image: np.ndarray, intensity: float = 0.02) -> np.ndarray:
    """Simulate ink bleeding effect."""
    if intensity <= 0:
        return image

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(image, kernel, iterations=1)

    result = cv2.addWeighted(image, 1 - intensity, dilated, intensity, 0)
    return result


def add_shadows(image: np.ndarray, shadow_intensity: float = 0.1) -> np.ndarray:
    """Add uneven lighting/shadow effects."""
    if shadow_intensity <= 0:
        return image

    h, w = image.shape[:2]

    gradient = np.linspace(0, shadow_intensity, w)
    gradient = np.tile(gradient, (h, 1))

    shadow = (1 - gradient) * 255
    shadow = np.stack([shadow] * 3, axis=-1)

    result = cv2.addWeighted(image, 1, shadow.astype(np.uint8), 0, 0)
    return result


def add_scan_artifacts(image: np.ndarray, artifact_level: str = "medium") -> np.ndarray:
    """Add various scan artifacts based on intensity level."""

    if artifact_level == "low":
        noise_sigma = 3.0
        blur_prob = 0.2
        rotation_max = 1.0
    elif artifact_level == "medium":
        noise_sigma = 5.0
        blur_prob = 0.3
        rotation_max = 2.0
    elif artifact_level == "high":
        noise_sigma = 10.0
        blur_prob = 0.5
        rotation_max = 3.0
    else:
        return image

    result = add_gaussian_noise(image, noise_sigma)
    result = add_blur(result, blur_prob)
    result = add_rotation(result, rotation_max)
    result = add_speckle_noise(result, intensity=0.01)

    if artifact_level in ("medium", "high"):
        result = add_ink_bleeding(result, intensity=0.02)

    if artifact_level == "high":
        result = add_shadows(result, shadow_intensity=0.1)

    return result


def generate_synthetic_drawing(
    form: GOSTForm,
    dpi: int,
    output_path: Path,
    canvas_size: tuple[int, int] = (3508, 2480),
    add_artifacts: bool = True,
    artifact_level: str = "medium",
    stamp_position: str = "bottom_right",
) -> dict:
    """
    Generate a synthetic technical drawing with a GOST stamp.

    Args:
        form: GOST form specification
        dpi: DPI for the scan
        output_path: Where to save the generated image
        canvas_size: Size of the canvas (width, height) - A4 @ 300 DPI default
        add_artifacts: Whether to add scan artifacts
        artifact_level: Artifact intensity (low, medium, high)
        stamp_position: Where to place the stamp

    Returns:
        Dictionary with metadata about the generated image
    """
    canvas_w, canvas_h = canvas_size

    stamp = create_stamp_grid(form, dpi, add_text=True)
    stamp_h, stamp_w = stamp.shape[:2]

    margin = 50
    x = min(margin, canvas_w - stamp_w - margin)
    y = min(margin, canvas_h - stamp_h - margin)

    if stamp_position == "bottom_right":
        x = max(margin, canvas_w - stamp_w - margin)
        y = max(margin, canvas_h - stamp_h - margin)
    elif stamp_position == "bottom_left":
        x = margin
        y = max(margin, canvas_h - stamp_h - margin)
    elif stamp_position == "top_right":
        x = max(margin, canvas_w - stamp_w - margin)
        y = margin
    elif stamp_position == "top_left":
        x = margin
        y = margin

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    stamp_h_final = min(stamp_h, canvas_h - y)
    stamp_w_final = min(stamp_w, canvas_w - x)
    canvas[y : y + stamp_h_final, x : x + stamp_w_final] = stamp[
        :stamp_h_final, :stamp_w_final
    ]

    if add_artifacts:
        canvas = add_scan_artifacts(canvas, artifact_level)

    cv2.imwrite(str(output_path), canvas)

    return {
        "image_path": str(output_path),
        "stamp_bbox": [x, y, stamp_w, stamp_h],
        "form_type": form.name,
        "dpi": dpi,
        "width_mm": form.width_mm,
        "height_mm": form.height_mm,
    }


def generate_test_dataset(
    output_dir: Path,
    num_samples: int = 10,
    dpi_values: list[int] = None,
    forms: list[str] = None,
    artifact_levels: list[str] = None,
    paper_size: str = "A4",
) -> list[dict]:
    """
    Generate a test dataset of synthetic drawings.

    Args:
        output_dir: Directory to save generated images
        num_samples: Number of images to generate
        dpi_values: List of DPI values to use
        forms: List of GOST forms to use
        artifact_levels: List of artifact levels
        paper_size: Paper size (A0, A1, A2, A3, A4)

    Returns:
        List of metadata dictionaries for ground truth
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dpi_values is None:
        dpi_values = [200, 300, 400]
    if forms is None:
        forms = ["FORM_3", "FORM_4", "FORM_5"]
    if artifact_levels is None:
        artifact_levels = ["low", "medium", "high"]

    metadata_list = []

    paper_sizes = {
        "A0": (841, 1189),
        "A1": (594, 841),
        "A2": (420, 594),
        "A3": (297, 420),
        "A4": (210, 297),
    }

    def mm_to_px(w_mm, h_mm, dpi):
        return (int(w_mm * dpi / 25.4), int(h_mm * dpi / 25.4))

    a_series_sizes = {}
    for size_name in ["A0", "A1", "A2", "A3", "A4"]:
        w_mm, h_mm = paper_sizes[size_name]
        a_series_sizes[size_name] = {
            dpi: mm_to_px(w_mm, h_mm, dpi) for dpi in dpi_values
        }

    if paper_size not in a_series_sizes:
        paper_size = "A4"

    canvas_sizes = a_series_sizes[paper_size]

    for i in range(num_samples):
        dpi = random.choice(dpi_values)
        form_name = random.choice(forms)
        form = GOST_FORMS[form_name]
        artifact_level = random.choice(artifact_levels)

        output_path = (
            output_dir / f"synthetic_{form_name}_{dpi}dpi_{artifact_level}_{i:03d}.png"
        )

        canvas_size = canvas_sizes.get(dpi, canvas_sizes.get(dpi_values[0], (2480, 3508)))

        meta = generate_synthetic_drawing(
            form=form,
            dpi=dpi,
            output_path=output_path,
            canvas_size=canvas_size,
            add_artifacts=True,
            artifact_level=artifact_level,
        )

        meta["text"] = "ИЗМ\nЛист\n№ докум.\nПодпись\nДата"
        meta["notes"] = (
            f"Generated: form={form_name}, dpi={dpi}, artifacts={artifact_level}"
        )

        metadata_list.append(meta)

        gt_path = output_dir / f"{output_path.stem}.json"
        import json

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)

        print(f"  Generated: {output_path.name}")

    return metadata_list


if __name__ == "__main__":
    output_dir = Path("ground_truth")
    print(f"Generating synthetic test dataset in {output_dir}...")
    metadata = generate_test_dataset(output_dir, num_samples=10)
    print(f"\nGenerated {len(metadata)} images with ground truth.")
