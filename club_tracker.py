"""
Трекинг булав в видео: HSV-сегментация + анализ траекторий полёта предметов.

Булавы художественной гимнастики обычно имеют яркую окраску (жёлтые, красные,
оранжевые, синие), что позволяет выделять их цветовой сегментацией в пространстве HSV.
Помимо классического подхода, модуль поддерживает кастомную ONNX-модель (например,
обученную YOLOv8): если путь к модели задан, детекция выполняется моделью, а HSV
используется как запасной вариант.

Анализируемые параметры полёта булав:
- высота броска каждой булавы (по максимальной точке траектории);
- симметрия броска (разница высот левой и правой булавы);
- одновременность (разница во времени достижения пика);
- точка и позиция ловли (x/y в момент, когда булава возвращается в зону рук).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --- Конфигурация HSV-сегментации по умолчанию ---
# Диапазоны подобраны для типичных ярких булав (жёлтый/оранжевый/красный).
DEFAULT_HSV_RANGES = [
    # Красный (два диапазона из-за разрыва оттенка на границе 0/180)
    ((0, 100, 80), (10, 255, 255)),
    ((160, 100, 80), (180, 255, 255)),
    # Оранжевый
    ((10, 100, 80), (25, 255, 255)),
    # Жёлтый
    ((25, 100, 80), (38, 255, 255)),
]

# Параметры морфологической очистки маски
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 250          # минимальная площадь контура булавы (в пикселях)
MAX_CONTOUR_AREA_RATIO = 0.08   # максимальная доля площади кадра для контура
MIN_DETECTION_CONFIDENCE = 0.35 # порог уверенности для ONNX-модели


@dataclass
class ClubDetection:
    """Обнаруженная булава в одном кадре."""
    x: float          # нормализованный центр x (0..1)
    y: float          # нормализованный центр y (0..1)
    area: float       # площадь в пикселях
    confidence: float = 1.0


@dataclass
class ClubTrajectory:
    """Траектория одной булавы за всё видео."""
    positions: List[Tuple[float, float]] = field(default_factory=list)  # (x, y)
    frames: List[int] = field(default_factory=list)  # индексы кадров детекций (для командного анализа)
    peak_y: Optional[float] = None     # минимальный y (самая высокая точка)
    peak_frame: Optional[int] = None   # номер кадра, когда достигнут пик
    catch_x: Optional[float] = None    # x в момент ловли
    catch_y: Optional[float] = None    # y в момент ловли


def _hsv_mask(hsv: np.ndarray) -> np.ndarray:
    """Строит бинарную маску по заданным HSV-диапазонам."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in DEFAULT_HSV_RANGES:
        part = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, part)
    return mask


def _contours_to_detections(mask: np.ndarray) -> List[ClubDetection]:
    """Превращает маску в список детекций булав (по контурам)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    frame_h, frame_w = mask.shape[:2]
    max_area = MAX_CONTOUR_AREA_RATIO * frame_h * frame_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        detections.append(
            ClubDetection(
                x=cx / frame_w,
                y=cy / frame_h,
                area=area,
            )
        )

    # Сортируем по x, чтобы «левая» и «правая» булава были стабильными
    detections.sort(key=lambda d: d.x)
    return detections


class ClubTracker:
    """
    Трекер булав: детекция на каждом кадре + сопоставление детекций
    с траекториями по расстоянию (простой двухкадровый трекинг).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hsv_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    ):
        self.model_path = model_path
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES
        self._net = None
        self._input_size = (640, 640)

        if model_path and os.path.exists(model_path):
            try:
                self._net = cv2.dnn.readNetFromONNX(model_path)
            except Exception as exc:  # pragma: no cover - зависит от окружения
                print(f"Не удалось загрузить ONNX-модель {model_path}: {exc}")
                self._net = None

    @property
    def uses_custom_model(self) -> bool:
        """Используется ли кастомная ONNX-модель."""
        return self._net is not None

    def _detect_with_model(self, frame_bgr: np.ndarray) -> List[ClubDetection]:
        """Детекция булав кастомной ONNX-моделью (формат выхода YOLOv8)."""
        blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, self._input_size, swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward()

        detections = []
        # YOLOv8 output shape: [1, 4+num_classes, num_anchors]
        out = outputs[0] if outputs.ndim == 3 else outputs
        if out.ndim != 3:
            return detections
        out = np.squeeze(out)
        if out.shape[0] < 5:
            return detections

        rows = out.shape[1]
        frame_h, frame_w = frame_bgr.shape[:2]
        for i in range(rows):
            row = out[:, i]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < MIN_DETECTION_CONFIDENCE:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # Проверяем, что бокс в пределах кадра
            if cx < 0 or cy < 0 or w <= 0 or h <= 0:
                continue
            x0 = max(0, cx - w / 2)
            y0 = max(0, cy - h / 2)
            x1 = min(frame_w, cx + w / 2)
            y1 = min(frame_h, cy + h / 2)
            detections.append(
                ClubDetection(
                    x=(x0 + x1) / 2 / frame_w,
                    y=(y0 + y1) / 2 / frame_h,
                    area=w * h,
                    confidence=confidence,
                )
            )

        detections.sort(key=lambda d: d.x)
        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[ClubDetection]:
        """Детектирует булавы на кадре (модель или HSV-сегментация)."""
        if self._net is not None:
            dets = self._detect_with_model(frame_bgr)
            if dets:
                return dets
            # fallback на HSV, если модель ничего не нашла
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = _hsv_mask(hsv)
        return _contours_to_detections(mask)

    def track(self, frames: List[np.ndarray]) -> List[ClubTrajectory]:
        """
        Обрабатывает последовательность кадров и возвращает траектории булав.

        frames: список кадров BGR.
        Возвращает список траекторий (до 2), отсортированных по средней позиции x.
        """
        trajectories: List[ClubTrajectory] = []

        for frame_idx, frame in enumerate(frames):
            detections = self.detect(frame)

            if not detections:
                # Если булавы не видны — траектории просто не получают новую точку
                continue

            # Если траекторий меньше, чем детекций — добавляем новые
            while len(trajectories) < len(detections):
                trajectories.append(ClubTrajectory())

            # Жадное сопоставление: для каждой детекции (по x) ищем ближайшую траекторию
            used = set()
            for det in detections:
                best_idx = None
                best_dist = float("inf")
                for idx, traj in enumerate(trajectories):
                    if idx in used or not traj.positions:
                        continue
                    last_x, last_y = traj.positions[-1]
                    dist = (det.x - last_x) ** 2 + (det.y - last_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                if best_idx is None:
                    # Нет подходящей траектории — создаём новую для этой детекции
                    # (или берём первую пустую)
                    empty_idx = next((i for i, t in enumerate(trajectories) if not t.positions), None)
                    if empty_idx is not None:
                        best_idx = empty_idx
                    else:
                        trajectories.append(ClubTrajectory())
                        best_idx = len(trajectories) - 1

                trajectories[best_idx].positions.append((det.x, det.y))
                trajectories[best_idx].frames.append(frame_idx)
                used.add(best_idx)

        # Постобработка: вычисляем пики и точку ловли для каждой траектории
        for traj in trajectories:
            if not traj.positions:
                continue
            ys = [p[1] for p in traj.positions]
            peak_idx = int(np.argmin(ys))
            traj.peak_y = float(np.min(ys))
            traj.peak_frame = peak_idx
            # Точка ловли — последняя точка траектории (возврат в зону рук)
            traj.catch_x, traj.catch_y = traj.positions[-1]

        # Оставляем только непустые траектории
        trajectories = [t for t in trajectories if t.positions]

        # Сортируем по среднему x, чтобы левая/правая были стабильны
        trajectories.sort(key=lambda t: sum(p[0] for p in t.positions) / len(t.positions))
        return trajectories


def analyze_club_flight(
    trajectories: List[ClubTrajectory],
    fps: float,
) -> dict:
    """
    Анализирует траектории булав и возвращает метрики полёта:

    {
        "clubs_detected": 0..N,
        "throw_height": { "left": ..., "right": ..., "max": ..., "min": ... },  # 0..1 (0 — верх кадра)
        "symmetry": { "height_diff": ..., "peak_frame_diff": ... },
        "simultaneity": { "peak_frame_diff": ..., "simultaneous": bool },
        "catch_position": { "left": {"x":..., "y":...}, "right": {...} },
    }
    """
    result = {
        "clubs_detected": len(trajectories),
        "throw_height": None,
        "symmetry": None,
        "simultaneity": None,
        "catch_position": None,
        "dropped": {},
    }

    if not trajectories:
        return result

    # Высота броска: y растёт вниз, поэтому МЕНЬШЕ y = ВЫШЕ бросок
    heights = {}
    for side, traj in zip(["left", "right"], trajectories):
        heights[side] = round(traj.peak_y, 3) if traj.peak_y is not None else None
    heights["max"] = round(min(h for h in heights.values() if h is not None), 3)
    heights["min"] = round(max(h for h in heights.values() if h is not None), 3)
    result["throw_height"] = heights

    # Критерий падения булавы (эвристика): траектория заканчивается у пола
    # (последний y близок к нижней границе кадра) и после пика больше не поднимается
    # на заметную высоту. Ловля, напротив, происходит в зоне рук (y заметно меньше 1).
    CLUB_DROP_Y_THRESHOLD = 0.85
    CLUB_DROP_REBOUND = 0.10
    result["dropped"] = {}
    for side, traj in zip(["left", "right"], trajectories):
        traj_dropped = None
        if traj.peak_y is not None and traj.positions:
            last_y = traj.positions[-1][1]
            ys_after_peak = [p[1] for p in traj.positions[traj.peak_frame:]]
            min_after_peak = min(ys_after_peak) if ys_after_peak else last_y
            rebound = traj.peak_y - min_after_peak  # насколько после пика снова поднималась
            traj_dropped = last_y >= CLUB_DROP_Y_THRESHOLD and rebound <= CLUB_DROP_REBOUND
        result["dropped"][side] = traj_dropped

    if len(trajectories) >= 2:
        left, right = trajectories[0], trajectories[1]
        # Симметрия высоты: разница пиков (0 — идеально)
        height_diff = None
        if left.peak_y is not None and right.peak_y is not None:
            height_diff = round(abs(left.peak_y - right.peak_y), 3)
        result["symmetry"] = {
            "height_diff": height_diff,
            "peak_frame_diff": None,
        }
        # Одновременность: разница номеров кадров достижения пика
        peak_frame_diff = None
        if left.peak_frame is not None and right.peak_frame is not None:
            peak_frame_diff = abs(left.peak_frame - right.peak_frame)
        result["symmetry"]["peak_frame_diff"] = peak_frame_diff
        result["simultaneity"] = {
            "peak_frame_diff": peak_frame_diff,
            "simultaneous": peak_frame_diff is not None and peak_frame_diff <= max(1, int(fps / 10)),
        }
        # Позиция ловли
        result["catch_position"] = {
            "left": {"x": left.catch_x, "y": left.catch_y},
            "right": {"x": right.catch_x, "y": right.catch_y},
        }

    return result


def assign_clubs_to_athletes(
    trajectories: List[ClubTrajectory],
    athlete_wrist_positions: List[List[Tuple[float, float]]],
) -> List[List[ClubTrajectory]]:
    """
    Связывает траектории булав с гимнастками по близости к запястьям.

    athlete_wrist_positions: список; для каждой гимнастки — список позиций запястий
    (нормализованные (x, y)) по кадрам. Размеры списков должны совпадать по кадрам:
    для каждого кадра число позиций запястий = число кадров, где эта гимнастка видна.

    Возвращает список списков траекторий: для каждой гимнастки — её булавы (до 2).
    """
    if not athlete_wrist_positions:
        return []

    # Для каждой гимнастки собираем все позиции запястий (по всем кадрам)
    wrist_points_per_athlete = []
    for wrists in athlete_wrist_positions:
        pts = [(x, y) for (x, y) in wrists]
        wrist_points_per_athlete.append(pts)

    # Для каждой траектории вычисляем среднюю позицию
    traj_means = []
    for traj in trajectories:
        if traj.positions:
            xs = [p[0] for p in traj.positions]
            ys = [p[1] for p in traj.positions]
            traj_means.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        else:
            traj_means.append((0.5, 0.5))

    # Жадное назначение: каждой траектории — ближайшая гимнастка по среднему расстоянию
    # до её запястий
    assignments = []  # для каждой траектории — индекс гимнастки
    used_athletes = set()
    # Сортируем траектории по «силе» (количеству точек) — сначала самые надёжные
    order = sorted(range(len(trajectories)), key=lambda i: len(trajectories[i].positions), reverse=True)
    for i in order:
        best_athlete = None
        best_dist = float("inf")
        for a_idx, wrist_pts in enumerate(wrist_points_per_athlete):
            if a_idx in used_athletes and len(trajectories) <= len(wrist_points_per_athlete):
                # Если гимнасток столько же, сколько траекторий — не переиспользуем
                continue
            if not wrist_pts:
                continue
            dist = min(
                (traj_means[i][0] - wx) ** 2 + (traj_means[i][1] - wy) ** 2
                for (wx, wy) in wrist_pts
            )
            if dist < best_dist:
                best_dist = dist
                best_athlete = a_idx
        if best_athlete is None:
            best_athlete = 0
        assignments.append((i, best_athlete))
        used_athletes.add(best_athlete)

    # Собираем результат: для каждой гимнастки — список её траекторий
    result = [[] for _ in range(len(athlete_wrist_positions))]
    for traj_idx, athlete_idx in assignments:
        result[athlete_idx].append(trajectories[traj_idx])

    # В каждой группе сортируем по x (левая/правая)
    for group in result:
        group.sort(key=lambda t: sum(p[0] for p in t.positions) / len(t.positions))

    return result