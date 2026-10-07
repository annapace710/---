"""
Обучение собственной модели детекции булав художественной гимнастики.

Модель: YOLOv8 (ultralytics). После обучения экспортируется в ONNX,
чтобы её можно было использовать в ClubTracker (club_tracker.py).

Требуемые зависимости:
    pip install ultralytics onnx

Подготовка датасета (формат YOLO):
    dataset/
      images/
        train/   # размеченные изображения
        val/     # валидационные изображения
      labels/
        train/   # txt-файлы: class_id x_center y_center width height (нормализованные)
        val/

Запуск:
    python train_club_detector.py --data dataset/data.yaml --epochs 50 --export-onnx

Результаты:
    runs/detect/train*/weights/best.pt  — лучшая модель PyTorch
    runs/detect/train*/weights/best.onnx — модель ONNX для ClubTracker
"""

import argparse
import os
from pathlib import Path

# Импортируем lazily, чтобы обычный запуск бэкенда не требовал ultralytics
try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение детектора булав (YOLOv8)")
    parser.add_argument("--data", required=True, help="Путь к data.yaml датасета YOLO")
    parser.add_argument("--epochs", type=int, default=50, help="Число эпох обучения")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения")
    parser.add_argument("--batch", type=int, default=16, help="Размер батча")
    parser.add_argument("--model", default="yolov8n.pt", help="Базовая модель YOLO (pretrained)")
    parser.add_argument("--device", default="cpu", help="Устройство: cpu, 0, 0,1 ...")
    parser.add_argument("--project", default="runs/detect", help="Директория результатов")
    parser.add_argument("--name", default="train", help="Имя эксперимента")
    parser.add_argument("--export-onnx", action="store_true", help="Экспортировать лучшую модель в ONNX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if YOLO is None:
        raise SystemExit(
            "Библиотека ultralytics не установлена. Выполните: pip install ultralytics onnx"
        )

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Файл датасета не найден: {data_path}")

    model = YOLO(args.model)

    print(f"Обучение YOLO на датасете {data_path} ({args.epochs} эпох)...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        # ultralytics может записать результат в папку с суффиксом
        candidates = sorted(Path(args.project).glob(f"{args.name}*/weights/best.pt"))
        if not candidates:
            raise SystemExit("Не найден файл best.pt после обучения")
        best_pt = candidates[-1]

    print(f"Лучшая модель: {best_pt}")

    if args.export_onnx:
        best_model = YOLO(str(best_pt))
        onnx_path = best_pt.with_suffix(".onnx")
        best_model.export(format="onnx", imgsz=args.imgsz)
        print(f"ONNX-модель экспортирована: {onnx_path}")
        print(
            "Использование в ClubTracker:\n"
            f"  tracker = ClubTracker(model_path='{onnx_path}')\n"
            "  (ONNX-модель имеет приоритет; при отсутствии детекций — fallback на HSV)"
        )


if __name__ == "__main__":
    main()