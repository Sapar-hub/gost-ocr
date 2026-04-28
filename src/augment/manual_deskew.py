"""Manual deskewing script for test images."""

from pathlib import Path

import cv2
import numpy as np

from gost_ocr.preprocessing import deskew_image, detect_skew_angle, rotate_image

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR = TEST_IMAGES_DIR / "deskewed"


def manual_deskew(image_path: Path, custom_angle: float = 1) -> None:
    """Deskew a single image and save the result."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load: {image_path}")
        return

    if custom_angle is not None:
        deskewed = rotate_image(image, custom_angle)
        detected_angle = detect_skew_angle(image)
        used_angle = custom_angle
    else:
        detected_angle = detect_skew_angle(image)
        deskewed, used_angle = deskew_image(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_deskewed.jpg"
    cv2.imwrite(str(output_path), deskewed)

    print(f"{image_path.name}:")
    print(f"  Detected angle: {detected_angle:.2f}°")
    print(f"  Applied angle: {used_angle:.2f}°")
    print(f"  Saved to: {output_path}")


def main():
    images_to_deskew = [
        "1761739613_reduktor-chervjachnyj_-vid-obschij.jpg",
        "1757253748_shtucer.jpg",
        "1760763253_shema-stropilnoj-sistemy.jpg",
        "2.png",
        "1761739668_vedomyj-val.jpg"
    ]

    for img_name in images_to_deskew:
        img_path = TEST_IMAGES_DIR / img_name
        if img_path.exists():
            manual_deskew(img_path)
        else:
            print(f"Not found: {img_path}")

    print(f"\nDeskewed images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
