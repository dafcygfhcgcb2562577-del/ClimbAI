from climb_ai.analyze import KNOWN_ENOUGH
from climb_ai.pose import read_frame_image, write_jpeg


def save_images(report, video, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    for number, spot in enumerate(report.spots, start=1):
        image = read_frame_image(video, spot.frame_index)
        if image is None:
            continue
        name = f"{'ok' if spot.done else 'miss'}_{number:02d}.jpg"
        write_jpeg(folder / name, image)
        spot.image = name


def summary(report):
    if report.climb_sec == (0.0, 0.0):
        return {"verdict": "нет данных", "headline": "Пролаз не разобран", "subtitle": report.note}

    if not report.checked:
        return {
            "verdict": "только выполненное",
            "headline": f"Техник распознано: {len(report.done)}",
            "subtitle": report.note,
        }

    if not report.spots:
        if report.known_share >= KNOWN_ENOUGH:
            return {
                "verdict": "техника не нужна",
                "headline": "Техника здесь не нужна была",
                "subtitle": report.note,
            }
        return {
            "verdict": "не с чем сравнить",
            "headline": "Сравнивать не с чем",
            "subtitle": report.note,
        }

    if not report.mistakes:
        return {"verdict": "хорошо", "headline": "Ошибок нет", "subtitle": ""}

    return {
        "verdict": "есть ошибки",
        "headline": _plural(len(report.mistakes), "ошибка", "ошибки", "ошибок"),
        "subtitle": "",
    }


def as_text(report):
    head = summary(report)
    lines = [head["headline"]]
    if head["subtitle"]:
        lines.append(head["subtitle"])

    if report.mistakes:
        lines += ["", "НЕ СДЕЛАНО"]
        lines += [f"  {_clock(s.at_sec)}  {s.title}" for s in report.mistakes]

    if report.correct:
        lines += ["", "СДЕЛАНО"]
        lines += [f"  {_clock(s.at_sec)}  {s.title}" for s in report.correct]

    if report.done and not report.checked:
        lines += ["", "ТЕХНИКИ НА ВИДЕО"]
        lines += [f"  {_clock(m.start_sec)}  {m.technique}" for m in report.done]

    return "\n".join(lines)


def describe(spot):
    return f"{_clock(spot.at_sec)} {spot.title}"


def _plural(count, one, few, many):
    if 11 <= count % 100 <= 14:
        return f"{count} {many}"
    last = count % 10
    if last == 1:
        return f"{count} {one}"
    if last in (2, 3, 4):
        return f"{count} {few}"
    return f"{count} {many}"


def _clock(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"
