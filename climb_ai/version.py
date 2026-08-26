import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "log", "-1", "--format=%h %cd", "--date=format:%d.%m %H:%M"],
            capture_output=True, text=True, timeout=3, encoding="utf-8", errors="replace",
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _newest_source_change():
    newest = 0.0
    for folder in ("climb_ai", "web"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            newest = max(newest, path.stat().st_mtime)
    return datetime.fromtimestamp(newest).strftime("%d.%m %H:%M") if newest else "?"


def describe():
    return {
        "commit": _git_commit(),
        "code_changed_at": _newest_source_change(),
        "server_started_at": datetime.now().strftime("%d.%m %H:%M"),
    }
