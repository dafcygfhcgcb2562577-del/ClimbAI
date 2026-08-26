import json
import logging
import re
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from climb_ai.analyze import analyze
from climb_ai.lessons import VIDEO_DIR_NAME, as_list as lessons
from climb_ai.pose import PoseEngine, UserError
from climb_ai.reference import Reference, has_reference, reference_path
from climb_ai.report import save_images, summary
from climb_ai.settings import settings
from climb_ai.version import describe as describe_version
from web.backend.jobs import JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("climb.web")

HERE = Path(__file__).parent
UPLOAD_CHUNK = 1024 * 1024


@asynccontextmanager
async def _lifespan(_app):
    url = f"http://{'127.0.0.1' if settings.host in ('0.0.0.0', '::') else settings.host}:{settings.port}/"
    logger.info("ClimbAI готов: %s", url)
    if settings.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    yield


app = FastAPI(title="ClimbAI", lifespan=_lifespan)
store = JobStore(settings.jobs)
templates = Jinja2Templates(directory=str(HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
app.mount("/files", StaticFiles(directory=str(settings.jobs)), name="files")

SAMPLES = Path(__file__).resolve().parents[2] / VIDEO_DIR_NAME
if SAMPLES.is_dir():
    app.mount("/образцы", StaticFiles(directory=str(SAMPLES)), name="образцы")


def _page(name, request, **context):
    version = describe_version()
    template = templates.get_template(name)
    return HTMLResponse(
        template.render(
            {
                "request": request,
                "метка": re.sub(r"\W", "", version.get("code_changed_at", "0")),
                **context,
            }
        )
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _page("index.html", request, есть_эталон=has_reference())


@app.get("/техники", response_class=HTMLResponse)
def techniques_page(request: Request):
    return _page("techniques.html", request, техники=lessons())


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Разбор не найден")
    return _page("job.html", request, job_id=job_id)


@app.get("/api/job/{job_id}")
def job_state(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Разбор не найден")
    return JSONResponse(
        {
            "статус": job.status,
            "шаг": job.stage,
            "прогресс": round(job.progress, 3),
            "готово": job.done,
            "ошибка": job.error,
            "результат": job.result,
        }
    )


@app.post("/api/analyze")
async def start_analysis(video: UploadFile = File(...)):
    if not settings.pose_model.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Нет модели позы: {settings.pose_model}. Положите pose_landmarker_full.task в artifacts.",
        )

    job = store.create(video.filename or "видео.mp4")
    try:
        saved = await _save_upload(video, store.folder(job.id))
    except HTTPException as error:
        store.fail(job, str(error.detail))
        raise
    except Exception as error:
        store.fail(job, f"Видео не загрузилось: {error}")
        raise
    store.start(job, lambda tell: _analyze_one(saved, store.folder(job.id), tell))
    return JSONResponse({"job_id": job.id})


def _analyze_one(video, folder, tell):
    reference = Reference.load(reference_path()) if has_reference() else None
    with PoseEngine(settings.pose_model, settings.pose_max_side) as engine:
        report = analyze(video, engine, reference, settings.sample_step, tell)

    save_images(report, video, folder)
    payload = report.as_dict()
    payload["итог"] = summary(report)
    (folder / "отчёт.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    video.unlink(missing_ok=True)
    return payload


async def _save_upload(video, folder):
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Ожидалось видео, получено {video.content_type}")
    target = folder / "видео.mp4"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with target.open("wb") as out:
        while chunk := await video.read(UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413, detail=f"Файл больше {settings.max_upload_mb} МБ"
                )
            out.write(chunk)
    return target


@app.exception_handler(UserError)
async def _user_error(_request: Request, error: UserError):
    return JSONResponse(status_code=400, content={"detail": str(error)})
