const форма = document.getElementById("форма");
const состояние = document.getElementById("состояние");

форма.addEventListener("submit", async (событие) => {
  событие.preventDefault();
  const файл = document.getElementById("видео").files[0];
  if (!файл) return;

  const кнопка = форма.querySelector("button");
  кнопка.disabled = true;
  состояние.textContent = "Загружаю видео…";

  const данные = new FormData();
  данные.append("video", файл);

  try {
    const ответ = await fetch("/api/analyze", { method: "POST", body: данные });
    const тело = await ответ.json();
    if (!ответ.ok) throw new Error(тело.detail || "не получилось загрузить");
    window.location.href = "/job/" + тело.job_id;
  } catch (ошибка) {
    состояние.textContent = "Ошибка: " + ошибка.message;
    состояние.classList.add("плохо");
    кнопка.disabled = false;
  }
});
