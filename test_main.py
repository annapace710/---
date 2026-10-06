import os
import sys
import unittest

# Добавляем корень проекта в путь, чтобы импортировать main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import validate_video_file, MAX_FILE_SIZE_BYTES, detect_errors_and_recommendations


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


if __name__ == "__main__":
    unittest.main()