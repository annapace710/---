// Параметры валидации загружаемого файла
const VALID_FORMATS = ["video/mp4", "video/webm", "video/x-m4v"];
const VALID_EXTENSIONS = [".mp4", ".webm", ".m4v"];
const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

// Ключ для хранения адреса API в localStorage
const API_URL_STORAGE_KEY = "gymnastics_api_url";
const DEFAULT_API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("analysis-form");
const apiUrlInput = document.getElementById("api-url");
const fileInput = document.getElementById("video-file");
const validationMessage = document.getElementById("validation-message");
const analyzeBtn = document.getElementById("analyze-btn");
const resultSection = document.getElementById("result");
const resultContent = document.getElementById("result-content");
const statusText = document.getElementById("status-text");

// Восстанавливаем сохранённый адрес API при загрузке страницы
const savedApiUrl = localStorage.getItem(API_URL_STORAGE_KEY);
if (savedApiUrl) {
  apiUrlInput.value = savedApiUrl;
}

/** Возвращает актуальный базовый URL API из поля ввода. */
function getApiBaseUrl() {
  const url = apiUrlInput.value.trim().replace(/\/+$/, "");
  return url || DEFAULT_API_URL;
}

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
  statusText.textContent = "";
}

/** Форматирует долю кадров с обнаруженной позой в проценты. */
function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

/** Формирует HTML с результатами анализа. */
function renderAnalysis(data) {
  const analysis = data.analysis || {};

  const items = [
    { label: "Всего кадров", value: analysis.total_frames ?? "—" },
    { label: "Частота кадров (FPS)", value: analysis.fps ?? "—" },
    { label: "Проанализировано кадров", value: analysis.frames_analyzed ?? "—" },
    { label: "Кадров с человеком", value: analysis.pose_detections ?? "—" },
    { label: "Присутствие в кадре", value: formatPercent(analysis.pose_presence) },
  ];

  const rowsHtml = items
    .map(
      (item) => `
      <div class="metric">
        <span class="metric-label">${item.label}</span>
        <span class="metric-value">${item.value}</span>
      </div>`
    )
    .join("");

  resultContent.innerHTML = `
    <div class="metrics-grid">${rowsHtml}</div>
  `;
}

/** Показывает сообщение об ошибке в блоке результата. */
function renderError(message) {
  resultContent.innerHTML = `<p class="error-message">⚠️ ${message}</p>`;
}

form.addEventListener("submit", async function (event) {
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
  const apiBaseUrl = getApiBaseUrl();

  // Сохраняем адрес API для следующего визита
  localStorage.setItem(API_URL_STORAGE_KEY, apiBaseUrl);

  // Готовим FormData с видео и выбранным упражнением
  const formData = new FormData();
  formData.append("video_file", file);
  formData.append("exercise", exercise);

  // Блокируем кнопку на время запроса
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Анализ…";
  resultSection.classList.remove("hidden");
  statusText.textContent = "⏳ Загружаем и анализируем видео…";

  try {
    const response = await fetch(`${apiBaseUrl}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || data.error || `Ошибка сервера (${response.status})`);
    }

    if (data.error) {
      renderError(data.error);
    } else {
      renderAnalysis(data);
      statusText.textContent = "✅ Анализ завершён.";
    }
  } catch (err) {
    console.error("Ошибка запроса:", err);
    renderError("Не удалось подключиться к серверу анализа. Проверьте адрес API и что бэкенд запущен.");
    statusText.textContent = "❌ Ошибка.";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Проанализировать";
  }
});