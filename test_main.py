import os
import sys
import unittest

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
)


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
        movement = self._movement(knee_angle={"left_avg_deg": 120.0, "right_avg_deg": 170.0})
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
        pose = _pose({23: (0.45, 0.5), 24: (0.55, 0.5), 27: (0.1, 1.0), 28: (0.9, 1.0)})
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


if __name__ == "__main__":
    unittest.main()