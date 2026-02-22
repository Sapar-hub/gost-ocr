from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from typing_extensions import Annotated

from .config import DEFAULT_IMAGES_PATH, OUTPUT_DIR
from .extraction import extract_text
from .localization import localize_images
from .preprocessing import load_images

app = typer.Typer(help="GOST OCR - извлечение метаданных из чертежей")


@app.command()
def preprocess(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    roi_position: Annotated[
        str,
        typer.Option(
            "--roi", help="Позиция ROI: bottom_right, bottom_left, top_right, top_left"
        ),
    ] = "bottom_right",
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """Предобработка изображений: deskew + flip + ROI"""
    flip_angles = [0, 90, 180, 270] if flip else [0]

    results = load_images(
        path, flip_angles=flip_angles, roi_position=roi_position, debug=debug
    )

    return results


@app.command(name="pipeline", help="Полный конвейер: предобработка, локализация и извлечение текста.")
def run_pipeline(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    roi_position: Annotated[
        str,
        typer.Option(
            "--roi", help="Позиция ROI: bottom_right, bottom_left, top_right, top_left"
        ),
    ] = "bottom_right",
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """
    Выполняет полный цикл обработки изображений:
    1. Предобработка (deskew, flip, ROI)
    2. Локализация штампа
    3. Извлечение текста (OCR)
    Результаты сохраняются в папку 'output'.
    """
    flip_angles = [0, 90, 180, 270] if flip else [0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Этап 1: Предобработка ===
    print("=== ЭТАП 1: Предобработка ===")
    preprocessed_images = load_images(
        path, flip_angles=flip_angles, roi_position=roi_position, debug=debug
    )

    # === Этап 2: Локализация ===
    print("\n=== ЭТАП 2: Локализация ===")
    localization_results = localize_images(preprocessed_images, draw_all=False, debug=debug)

    found_count = sum(1 for r in localization_results if r.stamp is not None)
    print(f"\nНайденные штампы: {found_count}/{len(localization_results)}")


    # === Этап 3: Извлечение текста ===
    print("\n=== ЭТАП 3: Извлечение текста ===")
    extraction_results = []
    for loc_res in localization_results:
        if loc_res.stamp:
            print(f"  Распознавание текста для: {loc_res.preprocessed.original_path.name}...")
            ext_res = extract_text(loc_res, debug=debug)
            if ext_res:
                extraction_results.append(ext_res)
                # Save result to JSON file
                output_filename = (
                    f"{loc_res.preprocessed.original_path.stem}_output.json"
                )
                output_path = OUTPUT_DIR / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(ext_res), f, ensure_ascii=False, indent=4)
                print(f"    -> Результат сохранен в: {output_path}")

    print(f"\n=== ИТОГО: извлечен текст из {len(extraction_results)} штампов ===")
    return extraction_results


@app.command(name="extract", help="Алиас для команды 'pipeline'.")
def extract_alias(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    roi_position: Annotated[
        str,
        typer.Option(
            "--roi", help="Позиция ROI: bottom_right, bottom_left, top_right, top_left"
        ),
    ] = "bottom_right",
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """Алиас для 'pipeline' для выполнения полного цикла обработки."""
    return run_pipeline(path, flip, roi_position, debug)


if __name__ == "__main__":
    app()
