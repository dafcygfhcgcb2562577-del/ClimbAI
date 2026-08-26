import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path = ENV_FILE):
    if not path.is_file():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


load_env_file()


def _path(name, default):
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _int(name, default):
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _str(name, default):
    return os.getenv(name, "").strip() or default


def _flag(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    artifacts: Path
    reference: Path
    uploads: Path
    jobs: Path
    pose_model: Path

    host: str
    port: int
    max_upload_mb: int
    open_browser: bool

    sample_step: int
    pose_max_side: int

    @classmethod
    def from_env(cls):
        artifacts = _path("CLIMB_ARTIFACTS_ROOT", PROJECT_ROOT / "artifacts")
        return cls(
            artifacts=artifacts,
            reference=_path("CLIMB_REFERENCE", artifacts / "эталон.json"),
            uploads=_path("CLIMB_UPLOADS_ROOT", artifacts / "uploads"),
            jobs=_path("CLIMB_WEB_JOBS_ROOT", artifacts / "web_jobs"),
            pose_model=_path("CLIMB_MODEL_PATH", artifacts / "pose_landmarker_full.task"),
            host=_str("CLIMB_API_HOST", "127.0.0.1"),
            port=_int("CLIMB_API_PORT", 8000),
            max_upload_mb=_int("CLIMB_MAX_UPLOAD_MB", 350),
            open_browser=_flag("CLIMB_OPEN_BROWSER"),
            sample_step=_int("CLIMB_SAMPLE_STEP", 5),
            pose_max_side=_int("CLIMB_POSE_MAX_SIDE", 960),
        )


settings = Settings.from_env()
