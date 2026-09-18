# rhythmic-gymnastics-assistant

Веб-приложение для анализа видео упражнений художественной гимнастики и выдачи рекомендаций по коррекции техники (риски с булавами)

## 📁 Структура проекта

- `index.html`, `style.css`, `script.js` — фронтенд (статический сайт)
- `main.py` — бэкенд на FastAPI (загрузка видео, анализ позы через MediaPipe)
- `requirements.txt` — зависимости Python
- `uploads/temp/` — временная директория для загруженных видео

## 🚀 Запуск бэкенда локально

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API будет доступен по адресу `http://127.0.0.1:8000`.
Интерактивная документация (Swagger UI): `http://127.0.0.1:8000/docs`.

## 📝 Примеры curl-команд

### Загрузка видео (успешный сценарий)

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "video_file=@/path/to/video.mp4" \
  -F "exercise=risk-buhavy"
```

Пример ответа:

```json
{
  "message": "Видео получено",
  "file": "a1b2c3d4e5_video.mp4",
  "exercise": "risk-buhavy",
  "analysis": {
    "total_frames": 150,
    "fps": 30.0,
    "frames_analyzed": 75,
    "pose_detections": 72,
    "pose_presence": 0.96,
    "movement": {
      "torso_tilt": {
        "avg_deg": 15.2,
        "max_deg": 82.4,
        "min_deg": 3.1
      },
      "wrist_height": {
        "avg_y": 0.42,
        "highest_y": 0.18,
        "reach_ratio": 0.82
      },
      "trajectory": {
        "total_distance": 3.42,
        "x_range": 0.6,
        "y_range": 0.8
      },
      "motion": {
        "avg_speed": 0.12,
        "max_speed": 0.45,
        "smoothness": 0.87,
        "jerk_count": 12
      }
    }
  }
}
```

### Загрузка с недопустимым форматом (ошибка валидации)

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "video_file=@/path/to/file.txt"
```

Ответ:

```json
{
  "error": "Недопустимый формат файла: .txt. Поддерживаются: .avi, .m4v, .mkv, .mov, .mp4, .webm"
}
```

### Загрузка без файла (ошибка)

```bash
curl -X POST http://127.0.0.1:8000/upload
```

Ответ — ошибка `422 Unprocessable Entity` от FastAPI.

## 🎬 Поддерживаемые форматы

MP4, WebM, M4V, AVI, MOV, MKV. Максимальный размер файла — 50 МБ.

## 📊 Метрики анализа движений

Бэкенд вычисляет следующие метрики на основе позы человека (MediaPipe Pose):

| Метрика | Описание |
|---------|----------|
| `torso_tilt` | Угол наклона корпуса от вертикали (средний, макс., мин.) |
| `wrist_height` | Высота запястий при подбросе (средняя, макс., досягаемость) |
| `trajectory` | Траектория центра масс (путь, размах X/Y) |
| `motion` | Скорость и плавность движения (средняя/макс. скорость, плавность, рывки) |

## 🔌 CORS

Фронтенд может обращаться к API со следующих источников:

- `https://annapace.sourcecraft.site`
- `http://localhost:8000`
- `http://127.0.0.1:8000`

Также разрешены обращения с туннелей Google Colab (Cloudflare/ngrok).