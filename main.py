import os
import uuid
import urllib.request
from fastapi import FastAPI, File, Form, UploadFile
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
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v"}
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

# Индексы ключевых точек MediaPipe Pose (33 точки)
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24
LANDMARK_LEFT_KNEE = 25
LANDMARK_RIGHT_KNEE = 26


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


def _point(landmarks, index: int) -> np.ndarray:
    """Возвращает координаты (x, y) ключевой точки."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Угол между двумя векторами в градусах."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_torso_tilt(landmarks) -> float:
    """
    Угол наклона корпуса от вертикали (в градусах).
    Используется середина плеч и середина таза.
    """
    shoulder_mid = (_point(landmarks, LANDMARK_LEFT_SHOULDER) + _point(landmarks, LANDMARK_RIGHT_SHOULDER)) / 2
    hip_mid = (_point(landmarks, LANDMARK_LEFT_HIP) + _point(landmarks, LANDMARK_RIGHT_HIP)) / 2

    torso_vec = shoulder_mid - hip_mid  # от таза к плечам
    vertical = np.array([0.0, -1.0])    # направление «вверх»

    return _angle_between(torso_vec, vertical)


def compute_wrist_height(landmarks) -> float:
    """
    Нормализованная высота запястий (среднее y левого и правого).
    Значение y растёт вниз: чем МЕНЬШЕ y, тем ВЫШЕ руки.
    """
    left_y = landmarks[LANDMARK_LEFT_WRIST].y
    right_y = landmarks[LANDMARK_RIGHT_WRIST].y
    return float((left_y + right_y) / 2)


def compute_center_of_mass(landmarks) -> np.ndarray:
    """
    Приблизительный центр масс тела: среднее всех видимых точек.
    Используется для отслеживания траектории движения.
    """
    visible = [
        np.array([lm.x, lm.y])
        for lm in landmarks
        if getattr(lm, "visibility", 1.0) > 0.5
    ]
    if not visible:
        return np.array([0.5, 0.5])
    return np.mean(visible, axis=0)


def compute_deep_metrics(
    torso_angles: list,
    wrist_heights: list,
    com_positions: list,
    fps: float,
) -> dict:
    """Агрегирует метрики углублённого анализа движений."""
    result = {}

    # 1. Угол наклона корпуса при кувырке
    if torso_angles:
        arr = np.array(torso_angles)
        result["torso_tilt"] = {
            "avg_deg": round(float(np.mean(arr)), 1),
            "max_deg": round(float(np.max(arr)), 1),
            "min_deg": round(float(np.min(arr)), 1),
        }
    else:
        result["torso_tilt"] = None

    # 2. Высота рук при подбросе булав
    if wrist_heights:
        arr = np.array(wrist_heights)
        highest_y = float(np.min(arr))          # самая верхняя точка запястий
        result["wrist_height"] = {
            "avg_y": round(float(np.mean(arr)), 3),
            "highest_y": round(highest_y, 3),
            # Доля высоты кадра, на которую поднимались руки (0..1)
            "reach_ratio": round(max(0.0, 1.0 - highest_y), 3),
        }
    else:
        result["wrist_height"] = None

    # 3. Траектория движения (центр масс)
    if len(com_positions) > 1:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        result["trajectory"] = {
            "total_distance": round(float(np.sum(distances)), 4),
            "x_range": round(float(np.ptp(com[:, 0])), 4),
            "y_range": round(float(np.ptp(com[:, 1])), 4),
        }
    else:
        result["trajectory"] = None

    # 4. Скорость и плавность движения
    if len(com_positions) > 2:
        com = np.array(com_positions)
        diffs = np.diff(com, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        speeds = distances * fps  # нормализованные единицы в секунду

        accelerations = np.diff(speeds)
        # Количество «рывков» — смен знака ускорения
        jerk_count = int(np.sum(np.abs(np.diff(np.sign(accelerations))) > 0))
        # Плавность: 1 при нулевом разбросе ускорений, стремится к 0 при рывках
        smoothness = round(1.0 / (1.0 + float(np.std(accelerations))), 4)

        result["motion"] = {
            "avg_speed": round(float(np.mean(speeds)), 4),
            "max_speed": round(float(np.max(speeds)), 4),
            "smoothness": smoothness,
            "jerk_count": jerk_count,
        }
    else:
        result["motion"] = None

    return result


# --- Детекция ошибок и генерация рекомендаций ---

# Эвристические пороги для оценки техники упражнений.
# Значения подобраны на основе типовых ошибок начинающих гимнасток
# и будут уточняться по мере накопления данных.
THRESHOLDS = {
    "risk-buhavy": {
        "min_tuck_tilt_deg": 60.0,   # минимальный макс. угол наклона при кувырке
        "min_reach_ratio": 0.30,     # минимальная высота подброса булав (0..1)
        "max_jerk_count": 15,        # допустимое число рывков при ловле
    },
    "jump": {
        "min_jump_y_range": 0.15,    # минимальный вертикальный размах центра масс
        "min_backbend_tilt_deg": 35.0,  # минимальный макс. наклон корпуса при прогибе
        "max_jerk_count": 15,        # допустимое число рывков при приземлении
    },
    "balance": {
        "max_com_x_range": 0.10,     # допустимый горизонтальный дрейф центра масс
        "max_avg_tilt_deg": 10.0,    # допустимый средний наклон корпуса
    },
    "rotation": {
        "max_avg_tilt_deg": 12.0,    # допустимый средний наклон оси вращения
        "max_com_x_range": 0.12,     # допустимое смещение оси вращения
        "max_jerk_count": 20,        # допустимое число рывков при вращении
    },
}

# Штрафы за ошибки (используются для расчёта общей оценки)
SEVERITY_PENALTY = {
    "critical": 25,
    "warning": 10,
}


def _fmt(value, suffix=""):
    """Форматирует число для текста рекомендации."""
    if value is None:
        return "—"
    return f"{value}{suffix}"


def detect_errors_and_recommendations(exercise: str, movement: dict) -> dict:
    """
    Детектирует ошибки техники и формирует рекомендации для конкретного
    упражнения на основе агрегированных метрик движения.

    Возвращает:
    {
        "overall_score": 0..100,
        "errors": [{"severity": "critical"|"warning", "message": "..."}],
        "recommendations": ["..."],
        "summary": "..."
    }
    """
    errors = []
    recommendations = []

    if exercise not in THRESHOLDS:
        return {
            "overall_score": None,
            "errors": [],
            "recommendations": [
                "Упражнение не распознано. Выберите упражнение из списка и повторите анализ."
            ],
            "summary": "Не удалось подобрать правила проверки для выбранного упражнения.",
        }

    thresholds = THRESHOLDS[exercise]

    torso = movement.get("torso_tilt") or {}
    wrist = movement.get("wrist_height") or {}
    trajectory = movement.get("trajectory") or {}
    motion = movement.get("motion") or {}

    # Если поза почти не была обнаружена — данных для проверки недостаточно
    if not torso and not wrist and not trajectory:
        return {
            "overall_score": None,
            "errors": [
                {"severity": "critical", "message": "Недостаточно данных о позе: человек не был стабильно обнаружен в кадре."}
            ],
            "recommendations": [
                "Снимите видео при хорошем освещении, в полный рост, без перекрытий.",
                "Убедитесь, что тело целиком попадает в кадр на протяжении всего упражнения.",
            ],
            "summary": "Анализ невозможен из-за отсутствия данных о позе.",
        }

    # --- Риск: ловля двух булав после кувырка ---
    if exercise == "risk-buhavy":
        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_tuck_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Кувырок выполнен недостаточно глубоко: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_tuck_tilt_deg']:.0f}°). "
                    "Булавы не успеют перелететь над кувырком."
                ),
            })
            recommendations.append(
                "Сильнее группируйтесь и активнее перекатывайтесь через плечо, чтобы корпус "
                "проходил под траекторией полёта булав."
            )

        reach_ratio = wrist.get("reach_ratio")
        if reach_ratio is not None and reach_ratio < thresholds["min_reach_ratio"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Подброс булав недостаточно высокий: руки поднялись лишь на "
                    f"{_fmt(round(reach_ratio * 100), '%')} высоты кадра "
                    f"(нужно не менее {thresholds['min_reach_ratio'] * 100:.0f}%)."
                ),
            })
            recommendations.append(
                "Подбрасывайте булавы энергичным движением от плеча, направляя их вверх, "
                "и добавляйте небольшой подъём на носки для увеличения высоты полёта."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Ловля булав жёсткая: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Встречайте булавы мягким движением кистей и амортизируйте ловлю "
                "сгибанием рук в локтях."
            )

    # --- Прыжок с прогибом назад ---
    elif exercise == "jump":
        y_range = trajectory.get("y_range")
        if y_range is not None and y_range < thresholds["min_jump_y_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Прыжок слишком низкий: вертикальный размах центра масс "
                    f"{_fmt(y_range)} (нужно не менее {thresholds['min_jump_y_range']}). "
                    "Прогиб не успевает раскрыться в полёте."
                ),
            })
            recommendations.append(
                "Отталкивайтесь активнее обеими ногами и вытягивайтесь вверх перед прогибом, "
                "чтобы увеличить время полёта."
            )

        max_tilt = torso.get("max_deg")
        if max_tilt is not None and max_tilt < thresholds["min_backbend_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Недостаточный прогиб назад: максимальный наклон корпуса "
                    f"{_fmt(max_tilt, '°')} (нужно не менее {thresholds['min_backbend_tilt_deg']:.0f}°)."
                ),
            })
            recommendations.append(
                "В верхней точке прыжка активнее прогибайтесь в грудном отделе и тяните "
                "руки назад, раскрывая плечи."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Приземление жёсткое: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Приземляйтесь мягко: с носка на полную стопу, слегка сгибая колени "
                "и сохраняя баланс."
            )

    # --- Равновесие на одной ноге ---
    elif exercise == "balance":
        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Раскачивание корпуса: центр масс смещается по горизонтали на "
                    f"{_fmt(x_range)} (допустимо не более {thresholds['max_com_x_range']}). "
                    "Равновесие теряется."
                ),
            })
            recommendations.append(
                "Зафиксируйте взгляд в одной точке, напрягите мышцы пресса и удерживайте "
                "опорную ногу неподвижной."
            )

        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Корпус наклонён: средний наклон {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Поза выглядит неустойчивой."
                ),
            })
            recommendations.append(
                "Держите спину вертикально, подтяните живот и не опускайте плечи."
            )

    # --- Поворот (пируэт) ---
    elif exercise == "rotation":
        avg_tilt = torso.get("avg_deg")
        if avg_tilt is not None and avg_tilt > thresholds["max_avg_tilt_deg"]:
            errors.append({
                "severity": "critical",
                "message": (
                    f"Ось вращения завалена: средний наклон корпуса {_fmt(avg_tilt, '°')} "
                    f"(допустимо не более {thresholds['max_avg_tilt_deg']:.0f}°). "
                    "Вращение получается неустойчивым."
                ),
            })
            recommendations.append(
                "Держите ось вращения строго вертикальной: тянитесь макушкой вверх "
                "и держите корпус собранным."
            )

        x_range = trajectory.get("x_range")
        if x_range is not None and x_range > thresholds["max_com_x_range"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Центр масс смещается по горизонтали на {_fmt(x_range)} "
                    f"(допустимо не более {thresholds['max_com_x_range']}). "
                    "Теряется ось вращения."
                ),
            })
            recommendations.append(
                "Перед началом поворота перенесите вес точно на опорную ногу "
                "и сохраняйте проекцию центра масс над стопой."
            )

        jerk_count = motion.get("jerk_count")
        if jerk_count is not None and jerk_count > thresholds["max_jerk_count"]:
            errors.append({
                "severity": "warning",
                "message": (
                    f"Вращение неравномерное: зафиксировано {jerk_count} рывков "
                    f"(допустимо не более {thresholds['max_jerk_count']})."
                ),
            })
            recommendations.append(
                "Работайте руками равномерно, не «замирайте» между витками — "
                "вращение должно быть непрерывным."
            )

    # --- Общая оценка и итоговая рекомендация ---
    if not errors:
        overall_score = 100
        summary = "Отличная техника! Ошибок не обнаружено, продолжайте в том же духе."
        recommendations.insert(
            0,
            "Отличная техника! Для закрепления результата повторите упражнение "
            "в медленном темпе и под контролем тренера.",
        )
    else:
        overall_score = max(
            0,
            100 - sum(SEVERITY_PENALTY[e["severity"]] for e in errors),
        )
        if overall_score >= 80:
            summary = "Техника близка к идеалу: есть небольшие недочёты, которые легко исправить."
        elif overall_score >= 50:
            summary = "Упражнение выполняется, но есть заметные ошибки — обратите внимание на рекомендации."
        else:
            summary = "Техника требует серьёзной доработки. Разберите упражнение по элементам с тренером."

    return {
        "overall_score": overall_score,
        "errors": errors,
        "recommendations": recommendations,
        "summary": summary,
    }


def analyze_video(video_path: str, exercise: str = "unknown") -> dict:
    """
    Анализирует видео с помощью MediaPipe PoseLandmarker (Tasks API):
    определяет наличие человека, собирает статистику, вычисляет
    метрики углублённого анализа движений и формирует
    рекомендации по технике для выбранного упражнения.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Не удалось открыть видеофайл"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Анализируем примерно 2 кадра в секунду для экономии ресурсов
    frame_interval = max(1, int(fps / 2))

    frames_analyzed = 0
    pose_detections = 0

    # Сырые данные для углублённого анализа
    torso_angles = []
    wrist_heights = []
    com_positions = []

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
                    landmarks = results.pose_landmarks[0]
                    torso_angles.append(compute_torso_tilt(landmarks))
                    wrist_heights.append(compute_wrist_height(landmarks))
                    com_positions.append(compute_center_of_mass(landmarks))

                frames_analyzed += 1

            frame_idx += 1

    cap.release()

    movement = compute_deep_metrics(torso_angles, wrist_heights, com_positions, fps)

    # Детекция ошибок и генерация рекомендаций для выбранного упражнения
    feedback = detect_errors_and_recommendations(exercise, movement)

    return {
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frames_analyzed": frames_analyzed,
        "pose_detections": pose_detections,
        "pose_presence": round(pose_detections / max(frames_analyzed, 1), 3),
        "movement": movement,
        "feedback": feedback,
    }


@app.post("/upload")
async def upload_video(
    video_file: UploadFile = File(...),
    exercise: str = Form("unknown"),
):
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
        analysis = analyze_video(save_path, exercise)
        return {
            "message": "Видео получено",
            "file": unique_name,
            "exercise": exercise,
            "analysis": analysis,
        }
    finally:
        # 4. Очистка временного файла после обработки
        if os.path.exists(save_path):
            os.remove(save_path)