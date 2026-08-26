import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime

logger = logging.getLogger("climb.jobs")


@dataclass
class Job:
    id: str
    status: str = "в очереди"
    stage: str = "ждёт очереди"
    progress: float = 0.0
    created_at: str = ""
    video_name: str = ""
    error: str | None = None
    result: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def done(self):
        return self.status in ("готово", "ошибка")


class JobStore:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs = {}
        self._saved = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="climb")
        self._close_unfinished()

    def _close_unfinished(self):
        for path in sorted(self.root.glob("*/job.json")):
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if saved.get("status") in ("готово", "ошибка"):
                continue
            saved["status"] = "ошибка"
            saved["stage"] = "не получилось"
            saved["progress"] = 1.0
            saved["error"] = "Сервер перезапустился, разбор не закончен. Загрузите видео заново."
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")

    def folder(self, job_id):
        return self.root / job_id

    def create(self, video_name):
        job = Job(
            id=uuid.uuid4().hex,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            video_name=video_name,
        )
        with self._lock:
            self._jobs[job.id] = job
        self.folder(job.id).mkdir(parents=True, exist_ok=True)
        self._save(job)
        return job

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        path = self.folder(job_id) / "job.json"
        if not path.is_file():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Job)}
        return Job(**{key: value for key, value in saved.items() if key in known})

    def fail(self, job, reason):
        job.error = reason[:300]
        self._update(job, status="ошибка", stage="не получилось", progress=1.0)

    def start(self, job, work):
        def run():
            self._update(job, status="считаю", stage="начинаю", progress=0.02)
            try:
                job.result = work(lambda stage, part: self._update(job, stage=stage, progress=part))
                self._update(job, status="готово", stage="готово", progress=1.0)
            except Exception as error:
                logger.error("разбор упал: %s", traceback.format_exc())
                job.error = str(error).strip().splitlines()[0][:300] or type(error).__name__
                self._update(job, status="ошибка", stage="не получилось", progress=1.0)

        self._pool.submit(run)

    def _update(self, job, **changes):
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
        stamp = (job.status, job.stage, round(job.progress, 2))
        if stamp != self._saved.get(job.id):
            self._saved[job.id] = stamp
            self._save(job)

    def _save(self, job):
        path = self.folder(job.id) / "job.json"
        path.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
