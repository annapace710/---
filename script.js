// Параметры валидации загружаемого файла
const VALID_FORMATS = ["video/mp4", "video/webm", "video/x-m4v"];
const VALID_EXTENSIONS = [".mp4", ".webm", ".m4v"];
const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

// База демо-рекомендаций по упражнениям.
// В реальной версии эти тексты будет формировать YandexGPT на основе переданного видео.
const DEMO_RECOMMENDATIONS = {
  "risk-buhavy": {
    title: "Риск: ловля двух булав после кувырка под ними",
    points: [
      "Убедитесь, что обе булавы подброшены синхронно и на достаточную высоту для выполнения кувырка.",
      "Держите корпус собранным при входе в кувырок — подбородок к груди, спина округлена.",
      "Траектория полёта булав должна проходить ровно над телом гимнастки — избегайте смещения в стороны.",
      "В момент выхода из кувырка сфокусируйтесь на визуальном контроле обеих булав перед ловлей.",
      "Ловля выполняется мягко, с амортизацией движения руками, чтобы не сбить ритм упражнения."
    ]
  },
  "jump": {
    title: "Прыжок с прогибом назад",
    points: [
      "Отталкивайтесь полностью через стопу для максимальной высоты прыжка.",
      "Колени сохраняйте вместе, стопы натянуты во время фазы полёта.",
      "Прогиб назад выполняйте из грудного отдела, а не из поясницы, чтобы избежать травм.",
      "Руки должны завершить движение в согласованной позиции для чистоты линий.",
      "Приземляйтесь мягко, через носок, сохраняя равновесие."
    ]
  },
  "balance": {
    title: "Равновесие на одной ноге",
    points: [
      "Зафиксируйте взгляд на неподвижной точке перед собой для удержания баланса.",
      "Следите за тем, чтобы опорная нога была полностью выпрямлена.",
      "Поднятая нога должна образовывать точную геометрическую линию без отклонений.",
      "Работайте над силой и стабильностью мышц корпуса.",
      "Не допускайте дрожания и колебаний рук — они должны быть зафиксированы."
    ]
  },
  "rotation": {
    title: "Поворот (пируэт)",
    points: [
      "Начинайте поворот с мощного толчка, перенося вес на опорную ногу.",
      "Держите ось вращения вертикальной — не наклоняйтесь вперёд или назад.",
      "Голова поворачивается быстрее корпуса, используйте технику 'точек' для контроля.",
      "Руки должны быть собраны и зафиксированы, чтобы не потерять ось.",
      "Завершайте поворот в устойчивом положении, точно контролируя остановку."
    ]
  }
};

const form = document.getElementById("analysis-form");
const fileInput = document.getElementById("video-file");
const validationMessage = document.getElementById("validation-message");
const analyzeBtn = document.getElementById("analyze-btn");
const resultSection = document.getElementById("result");
const resultContent = document.getElementById("result-content");
const demoNote = document.getElementById("demo-note");

/**
 * Проверяет загруженный файл: наличие, формат и размер.
 * Формат проверяется и по MIME-типу, и по расширению (для M4V MIME может быть разным).
 * Возвращает сообщение об ошибке или пустую строку при успехе.
 */
function validateFile(file) {
  if (!file) {
    return "Пожалуйста, загрузите видеофайл.";
  }

  const fileName = (file.name || "").toLowerCase();
  const hasValidMime = VALID_FORMATS.includes(file.type);
  const hasValidExtension = VALID_EXTENSIONS.some((ext) => fileName.endsWith(ext));

  if (!hasValidMime && !hasValidExtension) {
    return "Недопустимый формат файла. Поддерживаются только MP4, WebM и M4V.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `Файл слишком большой. Максимальный размер — ${MAX_FILE_SIZE_MB} МБ.`;
  }

  return "";
}

/** Очищает вывод результатов и сообщение валидации. */
function resetOutput() {
  resultSection.classList.add("hidden");
  resultContent.innerHTML = "";
  demoNote.textContent = "";
}

/** Формирует HTML-список рекомендаций для выбранного упражнения. */
function renderRecommendations(exerciseKey) {
  const data = DEMO_RECOMMENDATIONS[exerciseKey];
  if (!data) {
    return;
  }

  const listHtml = data.points
    .map((point) => `<li>${point}</li>`)
    .join("");

  resultContent.innerHTML = `
    <h3>${data.title}</h3>
    <ul>${listHtml}</ul>
  `;
  demoNote.textContent =
    "Это демонстрационные рекомендации. Реальный анализ видео с помощью YandexGPT будет добавлен позже.";

  resultSection.classList.remove("hidden");
}

form.addEventListener("submit", function (event) {
  event.preventDefault();

  resetOutput();
  validationMessage.textContent = "";

  const file = fileInput.files[0];
  const error = validateFile(file);

  if (error) {
    validationMessage.textContent = error;
    return;
  }

  const exercise = document.getElementById("exercise-select").value;
  renderRecommendations(exercise);
});