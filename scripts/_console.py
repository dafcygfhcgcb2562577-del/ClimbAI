import io
import sys


def setup():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        if getattr(stream, "encoding", "").lower().replace("-", "") == "utf8":
            continue
        wrapped = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)
        setattr(sys, name, wrapped)
