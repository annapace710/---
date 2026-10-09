"""
Генерация PDF-карточки тренировки (reportlab).

Карточка содержит:
- заголовок: упражнение, формат, дата, оценка техники (0–100);
- миниатюру видео (кадр из начала видео) и скриншоты моментов ошибок;
- на каждом скриншоте — обводка положения скелета в момент ошибки;
- список ошибок с severity, рекомендации и предлагаемое упражнение;
- динамику по сравнению с предыдущей тренировкой по тому же элементу
  (например, «Угол разгибания вырос на 5° по сравнению с 05.10»);
- тепловую карту позиций гимнасток по площадке (для группового формата);
- кликабельные таймкоды — ссылки, перескакивающие к моменту ошибки
  в исходном видео (веб-плеер поддерживает параметр ?t=<секунды>).
"""

from datetime import datetime

import storage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# Обозначения упражнений для карточки
EXERCISE_TITLES = {
    "risk-buhavy": "Риск: ловля двух булав после кувырка",
    "jump": "Прыжок с прогибом назад",
    "balance": "Равновесие на одной ноге",
    "rotation": "Поворот (пируэт)",
    "walkover-front": "Переворот вперёд",
}

# Метрики для блока динамики: ключ -> человекочитаемое название (метрики в градусах)
COMPARABLE_METRICS = {
    "knee_angle": "Угол разгибания колена",
    "leg_opening": "Угол раскрытия ног (шпагат)",
    "torso_tilt": "Наклон корпуса",
    "torso_tilt_signed": "Систематический наклон оси",
    "ankle_height": "Высота стоп при отрыве",
    "hip_height": "Высота таза",
}

# Ключевые соединения скелета для обводки на скриншоте (MediaPipe Pose)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # руки
    (11, 23), (12, 24), (23, 24),                        # корпус
    (23, 25), (25, 27), (24, 26), (26, 28),              # ноги
]

THUMB_W = 80 * mm
THUMB_H = 45 * mm


def _format_date(iso: str) -> str:
    """ISO-дата -> «05.10»."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d.%m")
    except (ValueError, TypeError):
        return "—"


def _draw_skeleton(c: canvas.Canvas, frame_path: str, landmarks) -> None:
    """Рисует скриншот ошибки с обводкой скелета в заданной позиции."""
    img = c.drawImage(
        frame_path, 0, 0, width=THUMB_W, height=THUMB_H,
        preserveAspectRatio=True, mask=None,
    )
    if not landmarks:
        return
    # drawImage возвращает объект; фактически координаты равны заданным,
    # но при preserveAspectRatio могут отличаться — используем реальные.
    iw, ih = img.drawWidth, img.drawHeight
    x0, y0 = (THUMB_W - iw) / 2, (THUMB_H - ih) / 2
    c.setStrokeColorRGB(1, 0.2, 0.2)
    c.setLineWidth(1.4)
    for a, b in POSE_CONNECTIONS:
        pa, pb = landmarks.get(a), landmarks.get(b)
        if not pa or not pb:
            continue
        c.line(
            x0 + pa["x"] * iw, y0 + (1 - pa["y"]) * ih,
            x0 + pb["x"] * iw, y0 + (1 - pb["y"]) * ih,
        )


def _wrap_text(c: canvas.Canvas, text: str, font: str, size: int, max_width: float):
    """Разбивает текст на строки, влезающие в max_width."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if c.stringWidth(candidate, font, size) > max_width:
            if line:
                lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _compare_metric_lines(current_movement: dict, previous_movement: dict, prev_date: str):
    """Строки динамики: изменение сопоставимых метрик в градусах."""
    lines = []
    for key, title in COMPARABLE_METRICS.items():
        cur = current_movement.get(key) or {}
        prev = previous_movement.get(key) or {}
        candidates = []
        for metric, label in (("avg_deg", "среднее"), ("max_deg", "максимальное")):
            a, b = cur.get(metric), prev.get(metric)
            if a is None or b is None or a == b:
                continue
            candidates.append((abs(a - b), metric, label, a - b))
        if not candidates:
            continue
        # Показываем самое заметное изменение метрики
        delta_abs, metric, label, delta = max(candidates, key=lambda t: t[0])
        delta = round(delta, 1)
        direction = "вырос" if delta > 0 else "снизился"
        lines.append(
            f"{title} ({label}) {direction} на {abs(delta)}° по сравнению с {prev_date}"
        )
        if len(lines) >= 5:
            return lines
    return lines


def heatmap_notes(analysis: dict):
    """Текстовые выводы по тепловой карте (где гимнастки сбиваются в кучу)."""
    hm = analysis.get("heatmap_stats") or {}
    notes = []
    peak = hm.get("peak_density")
    if peak is not None:
        notes.append(f"Максимальная плотность позиций: {peak} совпадений в одной зоне.")
    overlap = hm.get("overlap_ratio")
    if overlap is not None:
        pct = round(overlap * 100)
        if overlap > 0.5:
            notes.append(f"Гимнастки сбивались в кучу {pct}% времени — работайте над интервалами.")
        else:
            notes.append(f"Дисперсия позиций достаточная (сближение {pct}% времени).")
    return notes


def build_report_pdf(
    meta: dict,
    video_thumb_path=None,
    error_frames: dict = None,
    previous_meta: dict = None,
    output_path: str = "report.pdf",
) -> str:
    """
    Собирает PDF-карточку тренировки.

    meta: документ отчёта (см. storage.save_report);
    video_thumb_path: PNG-миниатюра видео или None;
    error_frames: {index: {"path": ..., "timestamp": ..., "landmarks": {...}}};
    previous_meta: метаданные предыдущего отчёта (для динамики) или None;
    output_path: куда записать PDF.
    """
    analysis = meta.get("analysis") or {}
    feedback = analysis.get("feedback") or {}
    exercise = meta.get("exercise") or ""
    fmt = meta.get("format") or "individual"
    score = feedback.get("overall_score")

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height

    # --- Заголовок ---
    c.setFillColorRGB(0.12, 0.35, 0.62)
    c.rect(0, height - 28 * mm, width, 28 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    title = EXERCISE_TITLES.get(exercise, exercise)
    for line in _wrap_text(c, title, "Helvetica-Bold", 15, width - 24 * mm)[:1]:
        c.drawString(12 * mm, height - 14 * mm, line)
    c.setFont("Helvetica", 9)
    date_str = _format_date(meta.get("created_at", ""))
    c.drawString(
        12 * mm, height - 22 * mm,
        f"Формат: {'групповое (5 гимнасток)' if fmt == 'group' else 'индивидуальное'}   •   "
        f"Дата: {date_str}   •   Отчёт: {str(meta.get('report_id', ''))[:8]}",
    )

    y = height - 34 * mm

    # --- Миниатюра видео + оценка ---
    if video_thumb_path:
        try:
            c.drawImage(video_thumb_path, 12 * mm, y - THUMB_H, width=THUMB_W, height=THUMB_H)
        except Exception:
            pass
    if score is not None:
        c.setFillColorRGB(0.12, 0.35, 0.62)
        c.setFont("Helvetica-Bold", 34)
        c.drawRightString(width - 12 * mm, y - 22 * mm, str(score))
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 12 * mm, y - 30 * mm, "из 100 — оценка техники")
    c.setFillColor(colors.black)

    y -= THUMB_H + 9 * mm

    # --- Ошибки: скриншот с обводкой, таймкод, рекомендация, упражнение ---
    errors = feedback.get("errors") or []
    recommendations = feedback.get("recommendations") or []
    if errors:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(12 * mm, y, "Ошибки и рекомендации")
        y -= 8 * mm

        for i, err in enumerate(errors):
            if y < 60 * mm:
                c.showPage()
                y = height - 20 * mm

            frame = (error_frames or {}).get(i) or {}
            frame_path = frame.get("path")

            if frame_path:
                try:
                    _draw_skeleton(c, frame_path, frame.get("landmarks"))
                except Exception:
                    pass
                y -= THUMB_H + 3 * mm

            # Таймкод: кликабельная ссылка ?t=<секунды> к моменту ошибки в видео
            ts = frame.get("timestamp")
            if ts is not None:
                video_url = analysis.get("video_url") or ""
                link = f"{video_url}?t={int(ts)}" if video_url else None
                label = f"{i + 1}. [{int(ts) // 60}:{int(ts) % 60:02d}] {err.get('message', '')}"
            else:
                link = None
                label = f"{i + 1}. {err.get('message', '')}"

            severity_color = (
                colors.HexColor("#c0392b") if err.get("severity") == "critical"
                else colors.HexColor("#b8860b")
            )
            c.setFillColor(severity_color)
            c.setFont("Helvetica-Bold", 10)
            text_y = y
            for line in _wrap_text(c, label, "Helvetica-Bold", 10, width - 28 * mm):
                c.drawString(14 * mm, text_y, line)
                text_y -= 5 * mm
            if link:
                c.linkURL(link, (14 * mm, y - 5 * mm, width - 14 * mm, text_y + 5 * mm))
            y = text_y - 1 * mm

            # Рекомендация
            rec = recommendations[i] if i < len(recommendations) else ""
            if rec:
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 9)
                for line in _wrap_text(c, rec, "Helvetica", 9, width - 30 * mm):
                    c.drawString(16 * mm, y, line)
                    y -= 4.5 * mm

            # Предлагаемое упражнение
            drill = err.get("suggested_drill")
            if drill:
                c.setFillColorRGB(0.15, 0.5, 0.25)
                c.setFont("Helvetica-Oblique", 9)
                for line in _wrap_text(c, f"Упражнение: {drill}", "Helvetica-Oblique", 9, width - 30 * mm):
                    c.drawString(16 * mm, y, line)
                    y -= 4.5 * mm

            c.setFillColor(colors.black)
            y -= 4 * mm

    # --- Командные замечания (групповой формат) ---
    team = analysis.get("team_feedback") or {}
    team_messages = team.get("messages") or []
    if team_messages and fmt == "group":
        if y < 70 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(12 * mm, y, "Командные замечания")
        y -= 7 * mm
        c.setFont("Helvetica", 10)
        for msg in team_messages:
            severity_color = (
                colors.HexColor("#c0392b") if msg.get("severity") == "critical"
                else colors.HexColor("#b8860b")
            )
            c.setFillColor(severity_color)
            for line in _wrap_text(c, f"• {msg.get('message', '')}", "Helvetica", 10, width - 30 * mm):
                c.drawString(16 * mm, y, line)
                y -= 5 * mm
        c.setFillColor(colors.black)

    # --- Динамика по сравнению с прошлой тренировкой ---
    if previous_meta:
        prev_analysis = previous_meta.get("analysis") or {}
        prev_fb = prev_analysis.get("feedback") or {}
        prev_date = _format_date(previous_meta.get("created_at", ""))
        if y < 70 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(12 * mm, y, f"Динамика (сравнение с тренировкой {prev_date})")
        y -= 7 * mm
        c.setFont("Helvetica", 10)

        prev_score = prev_fb.get("overall_score")
        if score is not None and prev_score is not None and score != prev_score:
            delta = score - prev_score
            direction = "выросла" if delta > 0 else "снизилась"
            c.drawString(
                16 * mm, y,
                f"Общая оценка {direction} на {abs(delta)} балл(ов) по сравнению с {prev_date}",
            )
            y -= 5.5 * mm

        # Тренд критических ошибок (pre/post-контроль по данным детектора)
        try:
            trend = storage.error_trend(exercise, fmt, k=2)
        except Exception:
            trend = None
        if trend and trend.get("sessions", 0) >= 4:
            early = trend.get("early_avg_critical")
            late = trend.get("late_avg_critical")
            delta_pct = trend.get("delta_pct")
            if delta_pct is not None and delta_pct < 0:
                trend_text = (
                    f"Критических ошибок: {early} → {late} "
                    f"({abs(delta_pct)}% меньше за {trend['sessions']} тренировок)"
                )
            elif delta_pct is not None:
                trend_text = (
                    f"Критических ошибок: {early} → {late} "
                    f"(+{delta_pct}% за {trend['sessions']} тренировок)"
                )
            else:
                trend_text = (
                    f"Критических ошибок: {early} → {late} "
                    f"за {trend['sessions']} тренировок"
                )
            c.drawString(16 * mm, y, trend_text)
            y -= 5 * mm

        for line in _compare_metric_lines(
            analysis.get("movement") or {},
            prev_analysis.get("movement") or {},
            prev_date,
        ):
            c.drawString(16 * mm, y, line)
            y -= 5 * mm

    # --- Тепловая карта (групповой формат) ---
    heatmap_path = analysis.get("heatmap")
    if heatmap_path and fmt == "group":
        if y < 70 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(12 * mm, y, "Тепловая карта перемещений (5 гимнасток)")
        y -= 8 * mm
        try:
            c.drawImage(heatmap_path, 12 * mm, y - THUMB_H, width=THUMB_W, height=THUMB_H)
            y -= THUMB_H + 4 * mm
        except Exception:
            c.setFont("Helvetica", 9)
            c.drawString(16 * mm, y, "Тепловая карта недоступна.")
            y -= 6 * mm
        c.setFont("Helvetica", 9)
        for note in heatmap_notes(analysis)[:3]:
            c.drawString(16 * mm, y, note)
            y -= 5 * mm

    c.save()
    return output_path
