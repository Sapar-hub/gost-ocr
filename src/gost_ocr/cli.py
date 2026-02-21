from __future__ import annotations

from pathlib import Path

import typer
from typing_extensions import Annotated

from .config import DEFAULT_IMAGES_PATH
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


@app.command()
def localize(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    draw_all: Annotated[
        bool, typer.Option("--draw-all", help="Рисовать все найденные контуры")
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
    """Полный конвейер: предобработка + локализация штампа"""
    flip_angles = [0, 90, 180, 270] if flip else [0]

    print("=== ЭТАП 1: Предобработка ===")
    preprocessed = load_images(
        path, flip_angles=flip_angles, roi_position=roi_position, debug=debug
    )

    print("\n=== ЭТАП 2: Локализация ===")
    results = localize_images(preprocessed, draw_all=draw_all, debug=debug)

    found_count = sum(1 for r in results if r.stamp is not None)
    print(f"\n=== ИТОГО: найдено штампов: {found_count}/{len(results)} ===")

    return results


@app.command()
def pipeline(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    draw_all: Annotated[
        bool, typer.Option("--draw-all", help="Рисовать все найденные контуры")
    ] = False,
    roi_position: Annotated[
        str, typer.Option("--roi", help="Позиция ROI")
    ] = "bottom_right",
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """Алиас для команды localize"""
    return localize(path, flip, draw_all, roi_position, debug)


if __name__ == "__main__":
    app()
