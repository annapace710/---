"""
Хранение отчётов тренировок.

Каждый отчёт — директория uploads/reports/<report_id>/, содержащая:
- meta.json     — метаданные и полный результат анализа (JSON);
- <video_file>  — исходное видео (для ссылок с таймкодами);
- frames/       — скриншоты моментов ошибок (PNG с обводкой).

Отчёты старше RETENTION_DAYS дней автоматически удаляются,
чтобы видео (до 50 МБ каждое) не занимали диск бесконечно.
"""

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "uploads", "reports")

# Сколько дней хранить отчёты вместе с видео
RETENTION_DAYS = 7

# report_id генерируется как uuid4().hex — 32 шестнадцатеричных символа.
# Проверка нужна, чтобы путь к отчёту нельзя было подделать (path traversal).
_REPORT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def create_report_id() -> str:
    """Генерирует уникальный идентификатор отчёта."""
    return uuid.uuid4().hex


def is_valid_report_id(report_id: str) -> bool:
    """Проверяет формат идентификатора отчёта (защита от path traversal)."""
    return bool(report_id) and bool(_REPORT_ID_RE.match(report_id))


def get_report_dir(report_id: str, reports_dir: str = REPORTS_DIR) -> str:
    """Возвращает директорию отчёта, не создавая её."""
    return os.path.join(reports_dir, report_id)


def save_report(
    report_id: str,
    analysis: dict,
    video_path: str,
    reports_dir: str = REPORTS_DIR,
) -> dict:
    """
    Сохраняет отчёт: переносит видео в директорию отчёта и пишет meta.json.

    analysis: результат analyze_video (должен содержать exercise/format);
    video_path: путь к временному видеофайлу (перемещается в отчёт).

    Возвращает документ метаданных.
    """
    if not is_valid_report_id(report_id):
        raise ValueError("Некорректный report_id")

    report_dir = get_report_dir(report_id, reports_dir)
    os.makedirs(report_dir, exist_ok=True)

    video_name = analysis.get("video_file") or f"{report_id}_video"
    video_target = os.path.join(report_dir, video_name)
    if video_path and os.path.exists(video_path):
        shutil.move(video_path, video_target)

    os.makedirs(os.path.join(report_dir, "frames"), exist_ok=True)

    doc = {
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exercise": analysis.get("exercise"),
        "format": analysis.get("format"),
        "video_file": video_name,
        "analysis": analysis,
    }
    with open(os.path.join(report_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return doc


def get_report(report_id: str, reports_dir: str = REPORTS_DIR):
    """Загружает метаданные отчёта или None, если отчёт не найден."""
    if not is_valid_report_id(report_id):
        return None
    meta_path = os.path.join(get_report_dir(report_id, reports_dir), "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def update_report_analysis(report_id: str, analysis: dict, reports_dir: str = REPORTS_DIR) -> None:
    """Обновляет блок analysis в meta.json отчёта."""
    doc = get_report(report_id, reports_dir)
    if doc is None:
        return
    doc["analysis"] = analysis
    meta_path = os.path.join(get_report_dir(report_id, reports_dir), "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


def find_previous_report(
    exercise: str,
    format_type: str,
    created_at: str,
    reports_dir: str = REPORTS_DIR,
):
    """
    Ищет предыдущий отчёт по тому же упражнению и формату (для динамики).
    Возвращает метаданные ближайшего по времени отчёта до created_at или None.
    """
    if not created_at or not os.path.isdir(reports_dir):
        return None

    best = None
    for entry in os.listdir(reports_dir):
        doc = get_report(entry, reports_dir)
        if not doc:
            continue
        if doc.get("exercise") != exercise or doc.get("format") != format_type:
            continue
        prev_created = doc.get("created_at") or ""
        if prev_created >= created_at:
            continue
        if best is None or prev_created > (best.get("created_at") or ""):
            best = doc
    return best


def cleanup_old_reports(reports_dir: str = REPORTS_DIR, retention_days: int = RETENTION_DAYS) -> int:
    """
    Удаляет отчёты старше retention_days дней. Возвращает число удалённых.
    Вызывается после сохран��ния нового отчёта.
    """
    if not os.path.isdir(reports_dir):
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    removed = 0
    for entry in os.listdir(reports_dir):
        report_dir = os.path.join(reports_dir, entry)
        if not os.path.isdir(report_dir):
            continue
        doc = get_report(entry, reports_dir)
        created_at = (doc or {}).get("created_at") or ""
        if created_at and created_at >= cutoff:
            continue
        try:
            shutil.rmtree(report_dir)
            removed += 1
        except OSError:
            continue
    return removed
