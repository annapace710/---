import os
import sys
import unittest

# Добавляем корень проекта в путь, чтобы импортировать main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import validate_video_file, MAX_FILE_SIZE_BYTES


class TestValidateVideoFile(unittest.TestCase):
    """Тесты для проверки валидации загружаемого видеофайла."""

    def test_valid_mp4(self):
        """Валидный MP4-файл не должен вызывать ошибку."""
        validate_video_file("video.mp4", 1024)
        # Если ошибка не возникла — тест пройден

    def test_valid_webm(self):
        """Валидный WebM-файл не должен вызывать ошибку."""
        validate_video_file("video.webm", 1024)

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


if __name__ == "__main__":
    unittest.main()