import os
import sys
import unittest

import cv2
import numpy as np

# Добавляем корень проекта в путь, чтобы импортировать main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    validate_video_file,
    MAX_FILE_SIZE_BYTES,
    detect_errors_and_recommendations,
    compute_knee_angle,
    compute_leg_opening_angle,
    compute_hip_height,
    compute_torso_tilt_signed,
    compute_ankle_y,
    compute_ankle_height,
    compute_arm_symmetry,
    compute_rotation_angle_world,
    select_target_tracks,
    AthleteTrack,
    track_athletes,
)
from club_tracker import ClubTracker, analyze_club_flight, ClubTrajectory, assign_clubs_to_athletes


class TestValidateVideoFile(unittest.TestCase):
    """Тесты для проверки валидации загружаемого видеофайла."""

    def test_valid_mp4(self):
        """Валидный MP4-файл не должен вызывать ошибку."""
        validate_video_file("video.mp4", 1024)
        # Если ошибка не возникла — тест пройден

    def test_valid_webm(self):
        """Валидный WebM-файл не должен вызывать ошибку."""
        validate_video_file("video.webm", 1024)

    def test_valid_m4v(self):
        """Валидный M4V-файл не должен вызывать ошибку."""
        validate_video_file("video.m4v", 1024)

    def test_valid_uppercase_extension(self):
        """Расширение в верхнем регистре (.MP4) должно быть принято."""
        validate_video_file("video.MP4", 1024)

    def test_invalid_extension(self):
        """Файл с недопустимым расширением должен вызывать ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_video_file("file.txt", 1024)
        self.assertIn("Недопустимый формат файла", str(ctx.exception))

    def test_invalid_extension_no_dot(self):
        """Файл без расширения должен вызывать ValueError."""
        with self.assertRaises(ValueError):
            validate_video_file("file", 1024)

    def test_file_too_large(self):
        """Файл больше максимального размера должен вызывать ValueError."""
        oversized = MAX_FILE_SIZE_BYTES + 1
        with self.assertRaises(ValueError) as ctx:
            validate_video_file("video.mp4", oversized)
        self.assertIn("Файл слишком большой", str(ctx.exception))

    def test_exact_max_size(self):
        """Файл точно максимального размера должен проходить валидацию."""
        validate_video_file("video.mp4", MAX_FILE_SIZE_BYTES)


class TestDetectErrorsAndRecommendations(unittest.TestCase):
    """Тесты для детекции ошибок и генерации рекомендаций."""

    def _movement(self, **overrides):
        """Базовый набор метрик без ошибок (идеальная техника)."""
        movement = {
            "torso_tilt": {"avg_deg": 5.0, "max_deg": 85.0, "min_deg": 2.0},
            "wrist_height": {"avg_y": 0.3, "highest_y": 0.1, "reach_ratio": 0.9},
            "trajectory": {"total_distance": 2.0, "x_range": 0.05, "y_range": 0.3},
            "motion": {"avg_speed": 0.1, "max_speed": 0.3, "smoothness": 0.9, "jerk_count": 5},
        }
        movement.update(overrides)
        return movement

    def test_perfect_risk_buhavy(self):
        """Идеальная техника риска с булавами не должна содержать ошибок."""
        result = detect_errors_and_recommendations("risk-buhavy", self._movement())
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["recommendations"])

    def test_shallow_tuck_risk_buhavy(self):
        """Мелкий кувырок должен определяться как критическая ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 10.0, "max_deg": 30.0, "min_deg": 5.0})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertLess(result["overall_score"], 100)

    def test_low_reach_risk_buhavy(self):
        """Низкий подброс булав должен определяться как критическая ошибка."""
        movement = self._movement(wrist_height={"avg_y": 0.6, "highest_y": 0.8, "reach_ratio": 0.1})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Подброс" in e["message"] for e in result["errors"]))

    def test_low_jump(self):
        """Низкий прыжок должен определяться как критическая ошибка."""
        movement = self._movement(trajectory={"total_distance": 1.0, "x_range": 0.05, "y_range": 0.05})
        result = detect_errors_and_recommendations("jump", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Прыжок слишком низкий" in e["message"] for e in result["errors"]))

    def test_no_backbend_jump(self):
        """Недостаточный прогиб в прыжке должен определяться как ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 5.0, "max_deg": 15.0, "min_deg": 2.0})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("прогиб" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_swaying_balance(self):
        """Раскачивание в равновесии должно определяться как критическая ошибка."""
        movement = self._movement(trajectory={"total_distance": 3.0, "x_range": 0.35, "y_range": 0.1})
        result = detect_errors_and_recommendations("balance", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Раскачивание" in e["message"] for e in result["errors"]))

    def test_tilted_rotation(self):
        """Заваленная ось вращения должна определяться как критическая ошибка."""
        movement = self._movement(torso_tilt={"avg_deg": 25.0, "max_deg": 40.0, "min_deg": 10.0})
        result = detect_errors_and_recommendations("rotation", movement)
        severities = [e["severity"] for e in result["errors"]]
        self.assertIn("critical", severities)
        self.assertTrue(any("Ось вращения" in e["message"] for e in result["errors"]))

    def test_unknown_exercise(self):
        """Неизвестное упражнение не должно давать ошибок, но должно дать рекомендацию."""
        result = detect_errors_and_recommendations("unknown-exercise", self._movement())
        self.assertIsNone(result["overall_score"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["recommendations"])

    def test_no_pose_data(self):
        """Отсутствие данных о позе должно давать критическую ошибку."""
        result = detect_errors_and_recommendations("balance", {})
        self.assertIsNone(result["overall_score"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["errors"][0]["severity"], "critical")

    def test_poor_grouping_risk_buhavy(self):
        """Недостаточная группировка (таз недостаточно низко) должна определяться как ошибка."""
        movement = self._movement(hip_height={"min_y": 0.2, "max_y": 0.4, "avg_y": 0.3})
        result = detect_errors_and_recommendations("risk-buhavy", movement)
        self.assertTrue(any("группировк" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_clubs_asymmetric_throw_risk_buhavy(self):
        """Асимметричный бросок булав должен определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.25, "peak_frame_diff": 1},
            "simultaneity": {"peak_frame_diff": 1, "simultaneous": True},
            "catch_position": {"left": {"x": 0.4, "y": 0.7}, "right": {"x": 0.6, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("разной высоте" in e["message"].lower() for e in result["errors"]))

    def test_clubs_not_simultaneous_risk_buhavy(self):
        """Неодновременный подброс булав должен определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.02, "peak_frame_diff": 8},
            "simultaneity": {"peak_frame_diff": 8, "simultaneous": False},
            "catch_position": {"left": {"x": 0.4, "y": 0.7}, "right": {"x": 0.6, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("не одновременно" in e["message"].lower() for e in result["errors"]))

    def test_clubs_catch_offset_risk_buhavy(self):
        """Ловля булавы со смещением от центра должна определяться как ошибка."""
        movement = self._movement()
        clubs = {
            "clubs_detected": 2,
            "throw_height": {"left": 0.2, "right": 0.2, "max": 0.2, "min": 0.2},
            "symmetry": {"height_diff": 0.02, "peak_frame_diff": 1},
            "simultaneity": {"peak_frame_diff": 1, "simultaneous": True},
            "catch_position": {"left": {"x": 0.1, "y": 0.7}, "right": {"x": 0.9, "y": 0.7}},
        }
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("смещена от центра" in e["message"].lower() for e in result["errors"]))

    def test_clubs_one_detected_risk_buhavy(self):
        """Одна обнаруженная булава — предупреждение о невозможности оценки симметрии."""
        movement = self._movement()
        clubs = {"clubs_detected": 1, "throw_height": None, "symmetry": None,
                 "simultaneity": None, "catch_position": None}
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("только одна" in e["message"].lower() for e in result["errors"]))

    def test_clubs_none_detected_risk_buhavy(self):
        """Булавы не обнаружены — предупреждение."""
        movement = self._movement()
        clubs = {"clubs_detected": 0, "throw_height": None, "symmetry": None,
                 "simultaneity": None, "catch_position": None}
        result = detect_errors_and_recommendations("risk-buhavy", movement, clubs)
        self.assertTrue(any("не обнаружены" in e["message"].lower() for e in result["errors"]))

    def test_clubs_none_param_risk_buhavy(self):
        """Если clubs=None (трекинг не выполнялся) — ошибок про булавы быть не должно."""
        movement = self._movement()
        result = detect_errors_and_recommendations("risk-buhavy", movement, None)
        self.assertFalse(any("булав" in e["message"].lower() for e in result["errors"]))

    def test_narrow_legs_jump(self):
        """Недостаточное раскрытие ног в прыжке должно определяться как ошибка."""
        movement = self._movement(leg_opening={"max_deg": 60.0, "avg_deg": 40.0})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("раскрыт" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_low_hip_lift_jump(self):
        """Недостаточный вертикальный отрыв (таз низко) должен определяться как ошибка."""
        movement = self._movement(hip_height={"min_y": 0.6, "max_y": 0.9, "avg_y": 0.75})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("отрыв" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_bent_support_leg_balance(self):
        """Согнутая опорная нога в равновесии должна определяться как ошибка."""
        # left_y(0.95) - right_y(0.35) = 0.6 > 0 => левая лодыжка ниже => опорная левая (120° < 150°)
        movement = self._movement(
            ankle_y={"left_avg_y": 0.95, "right_avg_y": 0.35, "avg_diff": 0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 170.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertTrue(any("опорн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_straight_support_leg_balance(self):
        """Выпрямленная опорная нога в равновесии не должна давать ошибку."""
        movement = self._movement(knee_angle={"left_avg_deg": 175.0, "right_avg_deg": 172.0})
        result = detect_errors_and_recommendations("balance", movement)
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))

    def test_systematic_tilt_rotation(self):
        """Систематический завал оси вращения должен определяться как ошибка."""
        movement = self._movement(torso_tilt_signed={"avg_deg": 18.0, "abs_max_deg": 25.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("завалена" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_symmetric_rotation(self):
        """Симметричное вращение не должно давать ошибку систематического завала."""
        movement = self._movement(torso_tilt_signed={"avg_deg": 3.0, "abs_max_deg": 8.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("систематически" in e["message"].lower() for e in result["errors"]))

    def test_few_turns_rotation(self):
        """Малое число витков в пируэте должно определяться как ошибка."""
        movement = self._movement(rotation_turns={"total_deg": 120.0, "turns": 0.33})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("витк" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_enough_turns_rotation(self):
        """Достаточное число витков не должно давать ошибку."""
        movement = self._movement(rotation_turns={"total_deg": 720.0, "turns": 2.0})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("витк" in e["message"].lower() for e in result["errors"]))

    def test_asymmetric_arms_rotation(self):
        """Асимметрия рук должна определяться как ошибка."""
        movement = self._movement(arm_symmetry={"avg_diff": 0.25, "max_diff": 0.4})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertTrue(any("асимметричн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_symmetric_arms_rotation(self):
        """Симметричные руки не должны давать ошибку."""
        movement = self._movement(arm_symmetry={"avg_diff": 0.03, "max_diff": 0.08})
        result = detect_errors_and_recommendations("rotation", movement)
        self.assertFalse(any("асимметричн" in e["message"].lower() for e in result["errors"]))

    def test_low_ankle_lift_jump(self):
        """Низкий отрыв стоп в прыжке должен определяться как ошибка."""
        movement = self._movement(ankle_height={"min_y": 0.85, "avg_y": 0.9})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertTrue(any("стоп" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(result["recommendations"])

    def test_high_ankle_lift_jump(self):
        """Высокий отрыв стоп в прыжке не должен давать ошибку."""
        movement = self._movement(ankle_height={"min_y": 0.5, "avg_y": 0.7})
        result = detect_errors_and_recommendations("jump", movement)
        self.assertFalse(any("стоп" in e["message"].lower() for e in result["errors"]))

    def test_bent_support_leg_by_ankle_balance(self):
        """Опорная нога определяется по высоте лодыжек; согнутая — ошибка."""
        movement = self._movement(
            ankle_y={"left_avg_y": 0.3, "right_avg_y": 0.9, "avg_diff": -0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        # Левая лодыжка выше -> опорная правая (175°), ошибки быть не должно
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))

    def test_bent_support_leg_left_balance(self):
        """Если опорная левая нога согнута — ошибка с указанием ноги."""
        # left_y(0.95) - right_y(0.35) = 0.6 > 0 => левая лодыжка ниже => опорная левая
        movement = self._movement(
            ankle_y={"left_avg_y": 0.95, "right_avg_y": 0.35, "avg_diff": 0.6},
            knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertTrue(any("опорн" in e["message"].lower() for e in result["errors"]))
        self.assertTrue(any("левая" in e["message"].lower() for e in result["errors"]))

    def test_straight_support_leg_right_balance(self):
        """Если опорная правая нога выпрямлена — ошибки нет."""
        # left_y(0.35) - right_y(0.95) = -0.6 < 0 => правая лодыжка ниже => опорная правая
        movement = self._movement(
            ankle_y={"left_avg_y": 0.35, "right_avg_y": 0.95, "avg_diff": -0.6},
            knee_angle={"left_avg_deg": 175.0, "right_avg_deg": 175.0},
        )
        result = detect_errors_and_recommendations("balance", movement)
        self.assertFalse(any("опорн" in e["message"].lower() for e in result["errors"]))


class _Landmark:
    """Простой объект-заглушка с координатами x, y (и visibility)."""

    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _pose(landmarks_dict):
    """Создаёт список из 33 ключевых точек, недостающие заполняет нулями."""
    pose = [_Landmark(0.0, 0.0) for _ in range(33)]
    for idx, (x, y) in landmarks_dict.items():
        pose[idx] = _Landmark(x, y)
    return pose


class TestExerciseSpecificFeatures(unittest.TestCase):
    """Тесты для exercise-специфичных признаков."""

    def test_knee_angle_straight(self):
        """Полностью выпрямленная нога должна давать угол ~180°."""
        # hip(0.5, 0.4), knee(0.5, 0.7), ankle(0.5, 1.0)
        pose = _pose({23: (0.5, 0.4), 25: (0.5, 0.7), 27: (0.5, 1.0)})
        angle = compute_knee_angle(pose, "left")
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_knee_angle_bent(self):
        """Согнутая нога должна давать угол заметно меньше 180°."""
        # hip(0.5, 0.4), knee(0.5, 0.7), ankle(0.3, 0.9)
        pose = _pose({23: (0.5, 0.4), 25: (0.5, 0.7), 27: (0.3, 0.9)})
        angle = compute_knee_angle(pose, "left")
        self.assertLess(angle, 160.0)

    def test_leg_opening_narrow(self):
        """Ноги вместе должны давать малый угол раскрытия."""
        # hip_mid(0.5, 0.5), обе стопы в одной точке под тазом
        pose = _pose({23: (0.45, 0.5), 24: (0.55, 0.5), 27: (0.5, 1.0), 28: (0.5, 1.0)})
        angle = compute_leg_opening_angle(pose)
        self.assertLess(angle, 30.0)

    def test_leg_opening_wide(self):
        """Широко раскрытые ноги (шпагат) должны давать угол близкий к 180°."""
        # Горизонтальный шпагат: таз по центру, стопы на том же уровне по горизонтали
        pose = _pose({23: (0.45, 0.5), 24: (0.55, 0.5), 27: (0.0, 0.5), 28: (1.0, 0.5)})
        angle = compute_leg_opening_angle(pose)
        self.assertGreater(angle, 150.0)

    def test_hip_height(self):
        """Высота таза — среднее y левого и правого бедра."""
        pose = _pose({23: (0.4, 0.3), 24: (0.6, 0.5)})
        self.assertAlmostEqual(compute_hip_height(pose), 0.4, delta=0.001)

    def test_torso_tilt_signed_zero(self):
        """Вертикальный корпус должен давать нулевой наклон."""
        pose = _pose({11: (0.4, 0.2), 12: (0.6, 0.2), 23: (0.4, 0.6), 24: (0.6, 0.6)})
        self.assertAlmostEqual(compute_torso_tilt_signed(pose), 0.0, delta=1.0)

    def test_torso_tilt_signed_positive(self):
        """Наклон корпуса влево должен давать положительный угол."""
        # плечи смещены влево относительно таза
        pose = _pose({11: (0.35, 0.2), 12: (0.45, 0.2), 23: (0.4, 0.6), 24: (0.6, 0.6)})
        self.assertGreater(compute_torso_tilt_signed(pose), 0.0)

    def test_ankle_y(self):
        """Высота лодыжки — просто y ключевой точки."""
        pose = _pose({27: (0.5, 0.95), 28: (0.5, 0.3)})
        self.assertAlmostEqual(compute_ankle_y(pose, "left"), 0.95, delta=0.001)
        self.assertAlmostEqual(compute_ankle_y(pose, "right"), 0.3, delta=0.001)

    def test_ankle_height(self):
        """Средняя высота стоп — среднее y лодыжек."""
        pose = _pose({27: (0.5, 0.4), 28: (0.5, 0.8)})
        self.assertAlmostEqual(compute_ankle_height(pose), 0.6, delta=0.001)

    def test_arm_symmetry(self):
        """Асимметрия рук — модуль разницы высот запястий."""
        pose = _pose({15: (0.3, 0.25), 16: (0.7, 0.65)})
        self.assertAlmostEqual(compute_arm_symmetry(pose), 0.4, delta=0.001)

    def test_arm_symmetry_zero(self):
        """Симметричные руки дают нулевую асимметрию."""
        pose = _pose({15: (0.3, 0.4), 16: (0.7, 0.4)})
        self.assertAlmostEqual(compute_arm_symmetry(pose), 0.0, delta=0.001)

    def test_rotation_angle_world(self):
        """Угол ориентации плечевой линии по world-координатам."""
        class WL:
            def __init__(self, x, z):
                self.x = x
                self.z = z

        # Плечи направлены вдоль X (dz = 0) -> угол 0°
        world = {11: WL(0.0, 0.0), 12: WL(1.0, 0.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), 0.0, delta=1.0)

        # Плечи направлены вдоль Z (dx = 0) -> угол 90°
        world = {11: WL(0.0, 0.0), 12: WL(0.0, 1.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), 90.0, delta=1.0)

    def test_rotation_angle_world_negative(self):
        """Поворот в другую сторону даёт отрицательный угол."""
        class WL:
            def __init__(self, x, z):
                self.x = x
                self.z = z

        world = {11: WL(0.0, 0.0), 12: WL(0.0, -1.0)}
        self.assertAlmostEqual(compute_rotation_angle_world(world), -90.0, delta=1.0)


class TestClubTracker(unittest.TestCase):
    """Тесты для трекинга булав и анализа траекторий."""

    def test_analyze_club_flight_empty(self):
        """Без траекторий метрики полёта пустые."""
        result = analyze_club_flight([], 30.0)
        self.assertEqual(result["clubs_detected"], 0)
        self.assertIsNone(result["throw_height"])
        self.assertIsNone(result["symmetry"])

    def test_analyze_club_flight_two_clubs(self):
        """Две траектории: высота, симметрия, одновременность, ловля."""
        left = ClubTrajectory()
        left.positions = [(0.3, 0.8), (0.3, 0.5), (0.3, 0.2)]
        left.peak_y = 0.2
        left.peak_frame = 2
        left.catch_x, left.catch_y = 0.3, 0.7

        right = ClubTrajectory()
        right.positions = [(0.7, 0.8), (0.7, 0.4), (0.7, 0.2)]
        right.peak_y = 0.2
        right.peak_frame = 2
        right.catch_x, right.catch_y = 0.7, 0.7

        result = analyze_club_flight([left, right], 30.0)
        self.assertEqual(result["clubs_detected"], 2)
        self.assertEqual(result["throw_height"]["max"], 0.2)
        self.assertEqual(result["symmetry"]["height_diff"], 0.0)
        self.assertTrue(result["simultaneity"]["simultaneous"])
        self.assertEqual(result["catch_position"]["left"]["x"], 0.3)

    def test_analyze_club_flight_asymmetric(self):
        """Асимметричный бросок: разная высота и неодновременность."""
        left = ClubTrajectory()
        left.positions = [(0.3, 0.8), (0.3, 0.1), (0.3, 0.6)]
        left.peak_y = 0.1
        left.peak_frame = 1
        left.catch_x, left.catch_y = 0.3, 0.6

        right = ClubTrajectory()
        right.positions = [(0.7, 0.8), (0.7, 0.5), (0.7, 0.7)]
        right.peak_y = 0.5
        right.peak_frame = 6
        right.catch_x, right.catch_y = 0.7, 0.7

        result = analyze_club_flight([left, right], 30.0)
        self.assertEqual(result["throw_height"]["max"], 0.1)
        self.assertEqual(result["symmetry"]["height_diff"], 0.4)
        self.assertEqual(result["simultaneity"]["peak_frame_diff"], 5)
        self.assertFalse(result["simultaneity"]["simultaneous"])

    def test_detect_hsv_yellow_club(self):
        """HSV-детекция: жёлтый объект на тёмном фоне обнаруживается."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Рисуем жёлтый круг (BGR: (0, 255, 255))
        cv2.circle(frame, (50, 50), 12, (0, 255, 255), -1)
        tracker = ClubTracker()
        detections = tracker.detect(frame)
        self.assertGreaterEqual(len(detections), 1)

    def test_detect_hsv_no_clubs(self):
        """Без ярких объектов детекций нет."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Тёмный фон — ничего не должно детектироваться
        tracker = ClubTracker()
        detections = tracker.detect(frame)
        self.assertEqual(len(detections), 0)

    def test_track_returns_trajectories(self):
        """Трекинг последовательности кадров возвращает траектории с пиками."""
        frames = []
        for i in range(10):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            y = 180 - i * 10  # булава летит вверх
            cv2.circle(frame, (50, y), 10, (0, 255, 255), -1)  # жёлтая
            frames.append(frame)

        tracker = ClubTracker()
        trajectories = tracker.track(frames)
        self.assertGreaterEqual(len(trajectories), 1)
        self.assertIsNotNone(trajectories[0].peak_y)


class TestAthleteTracking(unittest.TestCase):
    """Тесты для трекинга гимнасток и связывания булав."""

    def _pose_with(self, center_x, center_y, scale=0.1):
        """Создаёт скелет (33 точки), сосредоточенный вокруг центра."""
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0

        pts = []
        for i in range(33):
            pts.append(LM(center_x + (i % 5 - 2) * scale / 2,
                          center_y + (i % 7 - 3) * scale / 2))
        return pts

    def test_select_target_tracks_individual(self):
        """individual выбирает самый крупный трек."""
        t1 = AthleteTrack(index=0)
        t1.size_sum = 10.0
        t1.frames_seen = 1
        t2 = AthleteTrack(index=1)
        t2.size_sum = 30.0
        t2.frames_seen = 1
        selected = select_target_tracks([t1, t2], "individual")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].index, 1)

    def test_select_target_tracks_group(self):
        """group выбирает 5 самых крупных треков."""
        tracks = []
        for i in range(7):
            t = AthleteTrack(index=i)
            t.size_sum = float(i)
            t.frames_seen = 1
            tracks.append(t)
        selected = select_target_tracks(tracks, "group")
        self.assertEqual(len(selected), 5)
        # Самые крупные — индексы 6,5,4,3,2
        idxs = [t.index for t in selected]
        self.assertEqual(idxs, [6, 5, 4, 3, 2])

    def test_assign_clubs_to_athletes(self):
        """Булавы привязываются к ближайшей гимнастке по запястьям."""
        left = ClubTrajectory()
        left.positions = [(0.2, 0.3), (0.2, 0.2)]
        right = ClubTrajectory()
        right.positions = [(0.8, 0.3), (0.8, 0.2)]

        # Гимнастка 0 — слева (запястья ~0.2), гимнастка 1 — справа (~0.8)
        wrist_positions = [
            [(0.2, 0.4), (0.2, 0.35)],
            [(0.8, 0.4), (0.8, 0.35)],
        ]
        result = assign_clubs_to_athletes([left, right], wrist_positions)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)
        # Левая булава — гимнастке 0, правая — гимнастке 1
        self.assertAlmostEqual(result[0][0].positions[0][0], 0.2, delta=0.001)
        self.assertAlmostEqual(result[1][0].positions[0][0], 0.8, delta=0.001)

    def test_athlete_track_add_frame(self):
        """AthleteTrack накапливает ряды признаков по кадрам."""
        # Заглушка landmarks: объект с x, y, visibility
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.visibility = 1.0

        pts = []
        for i in range(33):
            pts.append(LM(0.1 + i * 0.02, 0.1 + (i % 3) * 0.02))
        track = AthleteTrack(index=0)
        track.add_frame(pts)
        self.assertEqual(track.frames_seen, 1)
        self.assertEqual(len(track.torso_angles), 1)
        self.assertEqual(len(track.wrist_heights), 1)
        self.assertIsNotNone(track.last_center)

    def test_track_athletes_two_people(self):
        """Два человека в кадре → два трека."""
        frames = [
            (None, [self._pose_with(0.3, 0.4), self._pose_with(0.7, 0.4)], None),
            (None, [self._pose_with(0.3, 0.41), self._pose_with(0.7, 0.42)], None),
        ]
        tracks = track_athletes(frames, 30.0)
        self.assertEqual(len(tracks), 2)


if __name__ == "__main__":
    unittest.main()