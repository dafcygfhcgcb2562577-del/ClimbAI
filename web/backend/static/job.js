const номер = document.body.dataset.job;
const ход = document.getElementById("ход");
const шаг = document.getElementById("шаг");
const заполнение = document.getElementById("заполнение");

function время(секунды) {
  const целых = Math.floor(секунды);
  return Math.floor(целых / 60) + ":" + String(целых % 60).padStart(2, "0");
}

function текстом(значение) {
  const узел = document.createElement("span");
  узел.textContent = значение === undefined || значение === null ? "" : String(значение);
  return узел.innerHTML;
}

function блок(родитель, разметка) {
  родитель.innerHTML = разметка;
  родитель.hidden = false;
}

function карточка(место) {
  const кадр = место.image
    ? `<img src="/files/${номер}/${encodeURIComponent(место.image)}" alt="" loading="lazy" />`
    : "";
  const названия = текстом((место.techniques || []).join(", "));
  return `<figure class="место">
            ${кадр}
            <figcaption>${названия}<span>${время(место.at_sec)}</span></figcaption>
          </figure>`;
}

const ЦВЕТ = {
  "есть ошибки": "плохо",
  "хорошо": "хорошо",
  "только выполненное": "хорошо",
  "техника не нужна": "хорошо",
};

function показать(отчёт) {
  ход.hidden = true;
  const итог = отчёт["итог"] || {};

  блок(
    document.getElementById("итог"),
    `<div class="карточка вердикт вердикт--${ЦВЕТ[итог["verdict"]] || "тихо"}">
       <h2>${текстом(итог["headline"] || "")}</h2>
       ${итог["subtitle"] ? `<p>${текстом(итог["subtitle"])}</p>` : ""}
     </div>`
  );

  const ошибки = отчёт["mistakes"] || [];
  if (ошибки.length) {
    блок(
      document.getElementById("ошибки"),
      `<h2 class="заголовок">Не сделано</h2>
       <div class="места">${ошибки.map(карточка).join("")}</div>`
    );
  }

  const верно = отчёт["correct"] || [];
  if (верно.length) {
    блок(
      document.getElementById("выполнено"),
      `<h2 class="заголовок">Сделано</h2>
       <div class="места">${верно.map(карточка).join("")}</div>`
    );
  }
}

async function спросить() {
  const ответ = await fetch("/api/job/" + номер);
  if (!ответ.ok) {
    шаг.textContent = "Разбор не найден.";
    return;
  }
  const состояние = await ответ.json();
  шаг.textContent = состояние["шаг"];
  заполнение.style.width = Math.round(состояние["прогресс"] * 100) + "%";

  if (!состояние["готово"]) {
    setTimeout(спросить, 1200);
    return;
  }
  if (состояние["ошибка"]) {
    ход.innerHTML = `<p class="плохо">Не получилось разобрать: ${текстом(состояние["ошибка"])}</p>`;
    return;
  }
  показать(состояние["результат"] || {});
}

спросить();
