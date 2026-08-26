import io
import tokenize
from pathlib import Path

from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP = {"venv", ".git", "__pycache__", "node_modules", ".ruff_cache", ".pytest_cache", "artifacts"}

KINDS = {".py", ".js", ".css", ".html", ".bat"}


def code_lines(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return sum(1 for line in text.splitlines() if line.strip())

    skip = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                skip.add(token.start[0])
            elif token.type == tokenize.STRING and token.line.strip().startswith(('"""', "'''")):
                skip.update(range(token.start[0], token.end[0] + 1))
    except tokenize.TokenError:
        pass
    return sum(1 for number, line in enumerate(text.splitlines(), 1) if line.strip() and number not in skip)


def main():
    setup()
    parts = {}
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in KINDS:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in SKIP for part in relative.parts):
            continue
        branch = relative.parts[0] if len(relative.parts) > 1 else "корень"
        parts[branch] = parts.get(branch, 0) + code_lines(path)

    print("Строк без комментариев и пустых:\n")
    for branch in sorted(parts, key=lambda name: -parts[name]):
        print("  %-12s %5d" % (branch, parts[branch]))
    print("  %-12s %5d" % ("ВСЕГО", sum(parts.values())))


if __name__ == "__main__":
    main()
