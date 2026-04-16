from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from typing import Annotated

from .config import DEFAULT_IMAGES_PATH, OUTPUT_DIR
from .evaluation import evaluate_batch, print_evaluation_report, save_evaluation_report
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
            "--roi",
            help="Позиция ROI: top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, full_page, corners (все углы)",
        ),
    ] = "bottom_right",
    dpi_roi: Annotated[
        bool,
        typer.Option(
            "--dpi-roi/--no-dpi-roi",
            help="DPI-based ROI calculation (по умолчанию выкл)",
        ),
    ] = False,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию вкл)",
        ),
    ] = True,
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """Предобработка изображений: deskew + flip + ROI"""
    flip_angles = [0, 90, 180, 270] if flip else [0]

    if roi_position == "corners":
        all_results = []
        corner_positions = ["bottom_right", "bottom_left", "top_right", "top_left"]
        for corner in corner_positions:
            results = load_images(
                path,
                flip_angles=flip_angles,
                roi_position=corner,
                dpi_roi=dpi_roi,
                debug=debug,
            )
            all_results.extend(results)
        return all_results
    else:
        results = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_roi,
            debug=debug,
        )
        return results


@app.command(
    name="pipeline",
    help="Полный конвейер: предобработка, локализация и извлечение текста.",
)
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
            "--roi",
            help="Позиция ROI: top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, full_page, corners (все углы)",
        ),
    ] = "bottom_right",
    dpi_roi: Annotated[
        bool,
        typer.Option(
            "--dpi-roi/--no-dpi-roi",
            help="DPI-based ROI calculation (по умолчанию выкл)",
        ),
    ] = False,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию вкл)",
        ),
    ] = True,
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

    # Handle corners mode
    if roi_position == "corners":
        corner_positions = ["bottom_right", "bottom_left", "top_right", "top_left"]
        all_preprocessed = []
        all_localization = []

        for corner in corner_positions:
            print(f"=== ROI corner: {corner} ===")
            preprocessed_images = load_images(
                path,
                flip_angles=flip_angles,
                roi_position=corner,
                dpi_roi=dpi_roi,
                debug=debug,
            )
            all_preprocessed.extend(preprocessed_images)

            loc_results = localize_images(
                preprocessed_images,
                draw_all=False,
                debug=debug,
                filter_by_size=filter_by_size,
            )
            all_localization.extend(loc_results)

        preprocessed_images = all_preprocessed
        localization_results = all_localization
    else:
        preprocessed_images = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_roi,
            debug=debug,
        )
        localization_results = localize_images(
            preprocessed_images,
            draw_all=False,
            debug=debug,
            filter_by_size=filter_by_size,
        )

    found_count = sum(1 for r in localization_results if r.stamp is not None)
    print(f"\nНайденные штампы: {found_count}/{len(localization_results)}")

    print("\n=== ЭТАП 3: Извлечение текста ===")
    extraction_results = []
    processed_images = set()

    for loc_res in localization_results:
        if loc_res.stamp:
            img_name = loc_res.preprocessed.original_path.name
            if img_name in processed_images:
                continue
            processed_images.add(img_name)

            print(f"  Распознавание текста для: {img_name}...")
            ext_res = extract_text(loc_res, debug=debug)
            if ext_res:
                extraction_results.append(ext_res)
                output_filename = (
                    f"{loc_res.preprocessed.original_path.stem}_output.json"
                )
                output_path = OUTPUT_DIR / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(ext_res), f, ensure_ascii=False, indent=4)
                print(f"    -> Результат сохранен в: {output_path}")

    print(f"\n=== ИТОГО: извлечен текст из {len(extraction_results)} штампов ===")
    return extraction_results


@app.command(name="localize", help="Локализация штампа (без OCR)")
def localize(
    path: Annotated[
        Path, typer.Argument(help="Путь к файлу или папке с изображениями (png/jpg)")
    ] = DEFAULT_IMAGES_PATH,
    flip: Annotated[
        bool, typer.Option("--flip", "-f", help="Пробовать все повороты (0/90/180/270)")
    ] = False,
    roi_position: Annotated[
        str,
        typer.Option(
            "--roi",
            help="Позиция ROI: top, bottom, left, right, top_left, top_right, bottom_left, bottom_right, full_page, corners (все углы)",
        ),
    ] = "bottom_right",
    dpi_roi: Annotated[
        bool,
        typer.Option(
            "--dpi-roi/--no-dpi-roi",
            help="DPI-based ROI calculation (по умолчанию выкл)",
        ),
    ] = False,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию вкл)",
        ),
    ] = True,
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Сохранять промежуточные результаты")
    ] = False,
):
    """
    Локализация штампа: предобработка + поиск контуров.
    Не запускает OCR - полезно для тестирования локализации.
    """
    flip_angles = [0, 90, 180, 270] if flip else [0]

    if roi_position == "corners":
        corner_positions = ["bottom_right", "bottom_left", "top_right", "top_left"]
        all_preprocessed = []
        all_localization = []

        for corner in corner_positions:
            print(f"=== ROI corner: {corner} ===")
            preprocessed_images = load_images(
                path,
                flip_angles=flip_angles,
                roi_position=corner,
                dpi_roi=dpi_roi,
                debug=debug,
            )
            all_preprocessed.extend(preprocessed_images)

            loc_results = localize_images(
                preprocessed_images,
                draw_all=False,
                debug=debug,
                filter_by_size=filter_by_size,
            )
            all_localization.extend(loc_results)

        localization_results = all_localization
    else:
        preprocessed_images = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_roi,
            debug=debug,
        )
        localization_results = localize_images(
            preprocessed_images,
            draw_all=False,
            debug=debug,
            filter_by_size=filter_by_size,
        )

    found_count = sum(1 for r in localization_results if r.stamp is not None)
    print(f"\n=== ИТОГО: найдено штампов {found_count}/{len(localization_results)} ===")
    return localization_results


@app.command(name="evaluate", help="Оценка качества по ground truth")
def evaluate(
    ground_truth_dir: Annotated[
        Path, typer.Argument(help="Путь к папке с ground truth JSON файлами")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Путь к папке с результатами OCR")
    ] = OUTPUT_DIR,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou-threshold", help="Минимальный IoU для успешной локализации"
        ),
    ] = 0.5,
    cer_threshold: Annotated[
        float,
        typer.Option(
            "--cer-threshold", help="Максимальный CER для успешного распознавания"
        ),
    ] = 0.1,
    wer_threshold: Annotated[
        float,
        typer.Option(
            "--wer-threshold", help="Максимальный WER для успешного распознавания"
        ),
    ] = 0.2,
    save_report: Annotated[
        bool, typer.Option("--save-report", "-s", help="Сохранить отчет в JSON файл")
    ] = False,
):
    """
    Оценивает качество работы OCR pipeline по сравнению с ground truth.

    Вычисляет метрики:
    - IoU (Intersection over Union) для локализации штампа
    - CER (Character Error Rate) для распознавания текста
    - WER (Word Error Rate) для распознавания текста
    """
    print("=== Оценка качества ===")
    print(f"Ground truth: {ground_truth_dir}")
    print(f"Output: {output_dir}")

    result = evaluate_batch(
        ground_truth_dir,
        output_dir,
        iou_threshold=iou_threshold,
        cer_threshold=cer_threshold,
        wer_threshold=wer_threshold,
    )

    print_evaluation_report(result)

    if save_report:
        report_path = Path("evaluation_report.json")
        save_evaluation_report(result, report_path)
        print(f"Отчет сохранен в: {report_path}")

    return result


if __name__ == "__main__":
    app()
