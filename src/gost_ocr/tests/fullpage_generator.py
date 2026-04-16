from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


GOST_FORMS = {
    "FORM_3": {
        "name": "Форма 3",
        "width_mm": 185,
        "height_mm": 55,
        "rows": 5,
        "cols": 9,
    },
    "FORM_4": {
        "name": "Форма 4",
        "width_mm": 185,
        "height_mm": 115,
        "rows": 8,
        "cols": 9,
    },
    "FORM_5": {
        "name": "Форма 5",
        "width_mm": 297,
        "height_mm": 55,
        "rows": 5,
        "cols": 15,
    },
}


def mm_to_pixels(mm: float, dpi: int) -> int:
    return int(mm * dpi / 25.4)


def get_font_path() -> str:
    """Get font path with Cyrillic support."""
    system_font_paths = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in system_font_paths:
        if Path(fp).exists():
            return fp
    return ""


def draw_russian_text(
    img_array: np.ndarray, text: str, position: tuple[int, int], font_size: int
) -> np.ndarray:
    """Draw Russian text on image using PIL."""
    font_path = get_font_path()
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(0, 0, 0))
    return np.array(img_pil)


def add_noise(image: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    if noise_level <= 0:
        return image
    h, w = image.shape[:2]
    noise = np.random.normal(0, 255 * noise_level, (h, w, 3))
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def add_blur(image: np.ndarray, blur_prob: float = 0.3) -> np.ndarray:
    if random.random() > blur_prob:
        return image
    return cv2.GaussianBlur(image, (3, 3), 0)


def add_rotation(image: np.ndarray, max_angle: float = 2.0) -> np.ndarray:
    if max_angle <= 0:
        return image
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def add_scan_artifacts(image: np.ndarray, artifact_level: str = "medium") -> np.ndarray:
    if artifact_level == "low":
        noise_sigma, blur_prob, rotation = 3.0, 0.2, 1.0
    elif artifact_level == "medium":
        noise_sigma, blur_prob, rotation = 5.0, 0.3, 2.0
    elif artifact_level == "high":
        noise_sigma, blur_prob, rotation = 10.0, 0.5, 3.0
    else:
        return image

    result = add_noise(image, noise_sigma / 255)
    result = add_blur(result, blur_prob)
    result = add_rotation(result, rotation)
    return result


def create_stamp_grid(form: dict, dpi: int, add_text: bool = True) -> np.ndarray:
    """Create a stamp grid image for a given GOST form."""
    w = mm_to_pixels(form["width_mm"], dpi)
    h = mm_to_pixels(form["height_mm"], dpi)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    cell_w = w // form["cols"]
    cell_h = h // form["rows"]
    line_color = (0, 0, 0)
    thickness = max(1, dpi // 100)

    for i in range(form["rows"] + 1):
        cv2.line(img, (0, i * cell_h), (w, i * cell_h), line_color, thickness)
    for j in range(form["cols"] + 1):
        cv2.line(img, (j * cell_w, 0), (j * cell_w, h), line_color, thickness)

    if add_text:
        font_size = max(28, dpi // 8)
        text_samples = ["ИЗМ", "Лист", "№ докум.", "Подпись", "Дата"]

        for i, text in enumerate(text_samples[: form["rows"]]):
            y = i * cell_h + cell_h // 2
            img = draw_russian_text(
                img, text, (cell_w // 4, y - font_size // 2), font_size
            )

        for j in range(min(form["cols"], 3)):
            x = j * cell_w + cell_w // 2 - 10
            img = draw_russian_text(
                img, str(j + 1), (x, cell_h // 2 - font_size // 2), font_size
            )

        for i in range(form["rows"]):
            for j in range(form["cols"]):
                if random.random() < 0.3:
                    fill_text = random.choice(["", "А", "Б", "В", "Г", "1", "2", "3"])
                    if fill_text:
                        x = j * cell_w + cell_w // 4
                        y = i * cell_h + cell_h // 2
                        img = draw_russian_text(
                            img, fill_text, (x, y - font_size // 2), font_size
                        )

    return img


def add_technical_drawing_content(
    img: np.ndarray, dpi: int, complexity: str = "medium"
) -> np.ndarray:
    """Add technical drawing-like content."""
    h, w = img.shape[:2]
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    line_color = (random.randint(100, 180),) * 3

    num_elements = {"low": 20, "medium": 50, "high": 100}.get(complexity, 50)

    for _ in range(num_elements):
        element = random.choice(["line", "circle", "rect"])
        if element == "line":
            x1, y1 = random.randint(50, w - 50), random.randint(50, h - 50)
            x2, y2 = random.randint(50, w - 50), random.randint(50, h - 50)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=max(1, dpi // 300))
        elif element == "circle":
            cx, cy, r = (
                random.randint(100, w - 100),
                random.randint(100, h - 100),
                random.randint(20, min(w, h) // 8),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line_color, width=2)
        elif element == "rect":
            x1, y1 = random.randint(50, w - 100), random.randint(50, h - 100)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            draw.rectangle([x1, y1, x2, y2], outline=line_color, width=2)

    return np.array(img_pil)


def add_dimensions_and_text(img: np.ndarray, dpi: int) -> np.ndarray:
    """Add dimension lines and technical text."""
    h, w = img.shape[:2]
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font_size = max(14, dpi // 20)
    font_path = get_font_path()
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except:
        font = ImageFont.load_default()

    for _ in range(random.randint(5, 15)):
        x, y = random.randint(50, w - 150), random.randint(50, h - 50)
        dim_text = f"{random.randint(10, 500)}"
        draw.text((x, y), dim_text, font=font, fill=(0, 0, 0))

        line_len = random.randint(30, 100)
        draw.line(
            [(x, y + font_size), (x + line_len, y + font_size)], fill=(0, 0, 0), width=1
        )
        draw.line(
            [(x, y + font_size - 5), (x, y + font_size + 5)], fill=(0, 0, 0), width=1
        )
        draw.line(
            [(x + line_len, y + font_size - 5), (x + line_len, y + font_size + 5)],
            fill=(0, 0, 0),
            width=1,
        )

    return np.array(img_pil)


def generate_full_page_synthetic(
    output_dir: Path,
    num_samples: int = 10,
    dpi_values: list[int] = None,
    form_types: list[str] = None,
    artifact_levels: list[str] = None,
    drawing_complexity: str = "medium",
    orientation: str = "random",
) -> list[dict]:
    """Generate synthetic full-page technical drawings with stamp."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dpi_values is None:
        dpi_values = [200, 300, 400]
    if form_types is None:
        form_types = ["FORM_3", "FORM_4", "FORM_5"]
    if artifact_levels is None:
        artifact_levels = ["low", "medium", "high"]

    metadata_list = []

    canvas_sizes = {
        200: (1754, 2480),
        300: (2480, 3508),
        400: (3508, 4961),
    }
    canvas_sizes_landscape = {
        200: (2480, 1754),
        300: (3508, 2480),
        400: (4961, 3508),
    }

    for i in range(num_samples):
        dpi = random.choice(dpi_values)
        form_name = random.choice(form_types)
        form = GOST_FORMS[form_name]
        artifact_level = random.choice(artifact_levels)

        # Determine orientation
        if orientation == "random":
            is_landscape = random.random() < 0.5
        elif orientation == "landscape":
            is_landscape = True
        else:
            is_landscape = False

        # Canvas size
        if is_landscape:
            canvas_w, canvas_h = canvas_sizes_landscape.get(
                dpi, canvas_sizes_landscape[300]
            )
        else:
            canvas_w, canvas_h = canvas_sizes.get(dpi, canvas_sizes[300])

        # Create blank canvas
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Add content
        canvas = add_technical_drawing_content(canvas, dpi, drawing_complexity)
        canvas = add_dimensions_and_text(canvas, dpi)

        # Create and place stamp
        stamp = create_stamp_grid(form, dpi, add_text=True)
        stamp_h, stamp_w = stamp.shape[:2]

        margin = 50
        # Random corner placement
        corner = random.choice(["bottom_right", "bottom_left", "top_right", "top_left"])

        if corner == "bottom_right":
            x = max(margin, canvas_w - stamp_w - margin)
            y = max(margin, canvas_h - stamp_h - margin)
        elif corner == "bottom_left":
            x = margin
            y = max(margin, canvas_h - stamp_h - margin)
        elif corner == "top_right":
            x = max(margin, canvas_w - stamp_w - margin)
            y = margin
        else:  # top_left
            x = margin
            y = margin

        # Place stamp
        stamp_h_final = min(stamp_h, canvas_h - y)
        stamp_w_final = min(stamp_w, canvas_w - x)
        canvas[y : y + stamp_h_final, x : x + stamp_w_final] = stamp[
            :stamp_h_final, :stamp_w_final
        ]

        # Add artifacts
        if artifact_level != "none":
            canvas = add_scan_artifacts(canvas, artifact_level)

        # Save image
        orientation_str = "landscape" if is_landscape else "portrait"
        output_path = (
            output_dir
            / f"fullpage_{form_name}_{dpi}dpi_{artifact_level}_{orientation_str}_{i:03d}.png"
        )
        cv2.imwrite(str(output_path), canvas)

        # Save ground truth JSON
        meta = {
            "image_path": str(output_path),
            "stamp_bbox": [x, y, stamp_w, stamp_h],
            "form_type": form["name"],
            "dpi": dpi,
            "orientation": orientation_str,
            "corner": corner,
            "width_mm": form["width_mm"],
            "height_mm": form["height_mm"],
            "text": "ИЗМ\nЛист\n№ докум.\nПодпись\nДата",
            "notes": f"Generated: form={form_name}, dpi={dpi}, artifacts={artifact_level}, orientation={orientation_str}, corner={corner}, full_page=True",
        }

        gt_path = output_dir / f"{output_path.stem}.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)

        metadata_list.append(meta)
        print(f"  Generated: {output_path.name}")

    return metadata_list


if __name__ == "__main__":
    output_dir = Path("ground_truth/fullpage")
    print(f"Generating full-page synthetic drawings in {output_dir}...")
    metadata = generate_full_page_synthetic(output_dir, num_samples=10)
    print(f"\nGenerated {len(metadata)} full-page images.")
