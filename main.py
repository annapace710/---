import os
import uuid
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2  # OpenCV
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

app = FastAPI()

# CORS: разрешаем обращения с фронтенда (SourceCraft Sites), локальной разработки
# и туннелей Google Colab (Cloudflare/ngrok)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://annapace.sourcecraft.site",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(.*\.trycloudflare\.com|.*\.ngrok-free\.app|.*\.ngrok\.io)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директория для временного хранения загруженных видео
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "temp")
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Гарантируем существование директории для загрузок
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MediaPipe Tasks API (новый API, совместим с Python 3.13) ---
# Модель PoseLandmarker скачивается автоматически при первом запуске
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def ensure_pose_model() -> None:
    """Скачивает модель PoseLandmarker, если она ещё не загружена."""
    if not os.path.exists(MODEL_PATH):
        print("Загрузка модели PoseLandmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Модель загружена.")


ensure_pose_model()


def validate_video_file(filename: str, content_length: int) -> None:
    """Проверяет формат и размер загружаемого видеофайла."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Недопустимый формат файла: {ext}. "
            f"Поддерживаются: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if content_length > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Файл слишком большой. Максимальный размер — {MAX_FILE_SIZE_MB} МБ."
        )


def analyze_video(video_path: str) -> dict:
    """
    Анализирует видео с помощью MediaPipe PoseLandmarker (Tasks API):
    определяет наличие человека в кадре и собирает базовую статистику.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Не удалось открыть видеофайл"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Анализируем примерно 2 кадра в секунду для экономии ресурсов
    frame_interval = max(1, int(fps / 2))

    frames_analyzed = 0
    pose_detections = 0

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_idx * 1000 / fps) if fps else 0
                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                if results.pose_landmarks:
                    pose_detections += 1
                frames_analyzed += 1

            frame_idx += 1

    cap.release()

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames_analyzed": frames_analyzed,
        "pose_detections": pose_detections,
        "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
    }


@app.post("/upload")
async def upload_video(video_file: UploadFile = File(...)):
    filename = video_file.filename or "video.mp4"

    # 1. Валидация файла
    try:
        validate_video_file(filename, video_file.size or 0)
    except ValueError as e:
        return {"error": str(e)}

    # 2. Сохранение файла во временную директорию
    unique_name = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        with open(save_path, "wb") as buffer:
            while chunk := await video_file.read(1024 * 1024):
                buffer.write(chunk)

        # 3. Анализ видео
        analysis = analyze_video(save_path)
        return {"message": "Видео получено", "file": unique_name, "analysis": analysis}
    finally:
        # 4. Очистка временного файла после обработки
        if os.path.exists(save_path):
            os.remove(save_path)