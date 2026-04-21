from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from typing import Annotated

from .config import DEFAULT_IMAGES_PATH, OUTPUT_DIR, DEBUG_EXTRACTION_DIR
from .evaluation import evaluate_batch, print_evaluation_report, save_evaluation_report
from .extraction import extract_text
from .localization import localize_images
from .preprocessing import load_images, normalize_dpi, detect_roi_type

app = typer.Typer(help="GOST OCR - извлечение метаданных из чертежей")


@app.callback()
def common_options(
    ctx: typer.Context,
    recursive: Annotated[
        bool, typer.Option("-r", "--recursive", help="Рекурсивно обрабатывать вложенные папки")
    ] = False,
    debug: Annotated[
        bool, typer.Option("-d", "--debug", help="Сохранять промежуточные результаты")
    ] = False,
):
    ctx.obj = {"recursive": recursive, "debug": debug}


def find_images_recursive(path: Path) -> list[Path]:
    """Find all image files recursively if enabled."""
    images = []
    if path.is_file():
        return [path] if path.suffix.lower() in [".png", ".jpg", ".jpeg"] else []

    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            images.append(p)
    return sorted(images)


@app.command()
def preprocess(
    ctx: typer.Context,
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
    ] = "auto",
    dpi: Annotated[
        str | None,
        typer.Option(
            "--dpi",
            help="DPI: auto, 200, 300, 400, 600",
        ),
    ] = None,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию выкл)",
        ),
    ] = False,
):
    """Предобработка изображений: deskew + flip + ROI"""
    debug = ctx.obj.get("debug", False) if ctx.obj else False
    recursive = ctx.obj.get("recursive", False) if ctx.obj else False

    if recursive:
        images = find_images_recursive(path)
        print(f"Рекурсивно найдено изображений: {len(images)}")

    flip_angles = [0, 90, 180, 270] if flip else [0]

    if roi_position == "auto":
        roi_position = detect_roi_type(path)
        print(f"Detected ROI position: {roi_position}")

    dpi_value = normalize_dpi(dpi)

    if roi_position == "corners":
        all_results = []
        corner_positions = ["bottom_right", "bottom_left", "top_right", "top_left"]
        for corner in corner_positions:
            results = load_images(
                path,
                flip_angles=flip_angles,
                roi_position=corner,
                dpi_roi=dpi_value,
                debug=debug,
            )
            all_results.extend(results)
        return all_results
    else:
        results = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_value,
            debug=debug,
        )
        return results


@app.command(
    name="pipeline",
    help="Полный конвейер: предобработка, локализация и извлечение текста.",
)
def run_pipeline(
    ctx: typer.Context,
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
    ] = "auto",
    dpi: Annotated[
        str | None,
        typer.Option(
            "--dpi",
            help="DPI: auto, 200, 300, 400, 600",
        ),
    ] = None,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию выкл)",
        ),
    ] = False,
):
    """
    Выполняет полный цикл обработки изображений:
    1. Предобработка (deskew, flip, ROI)
    2. Локализация штампа
    3. Извлечение текста (OCR)
    Результаты сохраняются в папку 'output'.
    """
    debug = ctx.obj.get("debug", False) if ctx.obj else False
    recursive = ctx.obj.get("recursive", False) if ctx.obj else False

    if recursive:
        images = find_images_recursive(path)
        print(f"Рекурсивно найдено изображений: {len(images)}")

    flip_angles = [0, 90, 180, 270] if flip else [0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if roi_position == "auto":
        roi_position = detect_roi_type(path)
        print(f"Detected ROI position: {roi_position}")

    dpi_value = normalize_dpi(dpi)

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
                dpi_roi=dpi_value,
                debug=debug,
                filter_by_size=filter_by_size,
            )
            all_preprocessed.extend(preprocessed_images)

            loc_results = localize_images(
                preprocessed_images,
                draw_all=False,
                debug=debug,
            )
            all_localization.extend(loc_results)

        preprocessed_images = all_preprocessed
        localization_results = all_localization
    else:
        preprocessed_images = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_value,
            debug=debug,
            filter_by_size=filter_by_size,
        )
        localization_results = localize_images(
            preprocessed_images,
            draw_all=False,
            debug=debug,
        )

    found_count = sum(1 for r in localization_results if r.stamp is not None)
    print(f"\nНайденные штампы: {found_count}/{len(localization_results)}")

    print("\n=== ЭТАП 3: Извлечение текста ===")
    extraction_results = []
    
    # Group localization results by original image path to find the most confident stamp per image
    grouped_localization_results: dict[Path, LocalizationResult] = {}
    for loc_res in localization_results:
        if loc_res.stamp:
            original_path = loc_res.preprocessed.original_path
            if original_path not in grouped_localization_results or \
               loc_res.stamp.confidence > grouped_localization_results[original_path].stamp.confidence:
                grouped_localization_results[original_path] = loc_res

    for original_path, loc_res in grouped_localization_results.items():
        img_name = original_path.name
        print(f"  Распознавание текста для: {img_name}...")
        ext_res = extract_text(loc_res, debug=debug)
        if ext_res:
            extraction_results.append(ext_res)
            
            # Existing JSON output to OUTPUT_DIR
            output_filename = (
                f"{loc_res.preprocessed.original_path.stem}_output.json"
            )
            output_path = OUTPUT_DIR / output_filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(ext_res), f, ensure_ascii=False, indent=4)
            print(f"    -> Результат сохранен в: {output_path}")

            # New TXT output to DEBUG_EXTRACTION_DIR if debug is True
            if debug:
                DEBUG_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
                txt_output_filename = (
                    f"{loc_res.preprocessed.original_path.stem}_extracted_text.txt"
                )
                txt_output_path = DEBUG_EXTRACTION_DIR / txt_output_filename

                with open(txt_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Full Text:\n{ext_res.full_text}\n\n")
                    f.write("Text Blocks (text, confidence):\n")
                    for block in ext_res.text_blocks:
                        f.write(
                            f"  - Text: '{block.text}', Confidence: {block.confidence:.2f}\n"
                        )
                print(f"    -> Debug text saved to: {txt_output_path}")

    print(f"\n=== ИТОГО: извлечен текст из {len(extraction_results)} штампов ===")
    return extraction_results


@app.command(name="localize", help="Локализация штампа (без OCR)")
def localize(
    ctx: typer.Context,
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
    ] = "auto",
    dpi: Annotated[
        str | None,
        typer.Option(
            "--dpi",
            help="DPI: auto, 200, 300, 400, 600",
        ),
    ] = None,
    filter_by_size: Annotated[
        bool,
        typer.Option(
            "--filter-by-size/--no-filter-by-size",
            help="Filter stamps by size config (по умолчанию выкл)",
        ),
    ] = False,
):
    """
    Локализация штампа: предобработка + поиск контуров.
    Не запускает OCR - полезно для тестирования локализации.
    """
    debug = ctx.obj.get("debug", False) if ctx.obj else False
    recursive = ctx.obj.get("recursive", False) if ctx.obj else False

    if recursive:
        images = find_images_recursive(path)
        print(f"Рекурсивно найдено изображений: {len(images)}")

    flip_angles = [0, 90, 180, 270] if flip else [0]

    if roi_position == "auto":
        roi_position = detect_roi_type(path)
        print(f"Detected ROI position: {roi_position}")

    dpi_value = normalize_dpi(dpi)

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
                dpi_roi=dpi_value,
                debug=debug,
                filter_by_size=filter_by_size,
            )
            all_preprocessed.extend(preprocessed_images)

            loc_results = localize_images(
                preprocessed_images,
                draw_all=False,
                debug=debug,
            )
            all_localization.extend(loc_results)

        localization_results = all_localization
    else:
        preprocessed_images = load_images(
            path,
            flip_angles=flip_angles,
            roi_position=roi_position,
            dpi_roi=dpi_value,
            debug=debug,
            filter_by_size=filter_by_size,
        )
        localization_results = localize_images(
            preprocessed_images,
            draw_all=False,
            debug=debug,
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