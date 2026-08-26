import os

import uvicorn

from climb_ai.settings import settings

if __name__ == "__main__":
    reload = os.getenv("CLIMB_DEV_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
    if reload:
        print("Режим разработки: сервер перезапускается сам при изменении файлов.")
    uvicorn.run(
        "web.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        reload_dirs=["climb_ai", "web"] if reload else None,
        log_level="info",
    )
