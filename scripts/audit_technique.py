import sys
from pathlib import Path

import cv2
import numpy as np

from climb_ai.pose import PoseEngine, read_poses, video_aspect
from climb_ai.settings import settings
from climb_ai.technique import TECHNIQUES, Pose, detect, score_all
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

СПРАВКА = """Проверка правил распознавания техники: на что они срабатывают.

    python -m scripts.audit_technique poses    разобрать видео датасета в кэш поз
    python -m scripts.audit_technique where    распределение техник по кадрам
    python -m scripts.audit_technique geom     разброс геометрии позы
    python -m scripts.audit_technique top Скручивание   картинка с лучшими кадрами"""

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
DATASET = Path(r"C:\Users\user\Desktop\база данных")


def _videos_in_dataset():
    found = []
    for folder in sorted(p for p in DATASET.iterdir() if p.is_dir() and p.name.startswith("p")):
        for route in ("green", "orange"):
            video = folder / f"{route}.mp4"
            if video.exists():
                found.append((folder.name, route, video))
    return found


def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    with PoseEngine(settings.pose_model, settings.pose_max_side) as engine:
        for participant, route, video in _videos_in_dataset():
            out = CACHE / f"{participant}_{route}.npz"
            if out.exists():
                continue
            print(f"{participant}/{route} ...", flush=True)
            frames = read_poses(video, engine, settings.sample_step)
            np.savez_compressed(
                out,
                indexes=np.array([f.index for f in frames], dtype=np.int32),
                seconds=np.array([f.second for f in frames], dtype=np.float32),
                points=np.array([f.points for f in frames], dtype=np.float32),
                camera=np.array([f.camera_y for f in frames], dtype=np.float32),
                visible=np.array([f.visible for f in frames], dtype=np.float32),
                шаг=np.array(settings.sample_step),
                сторона=np.array(settings.pose_max_side),
                аспект=np.array(video_aspect(video)),
            )
            print(f"  поз найдено: {len(frames)}")


def cached():
    out = []
    for path in sorted(CACHE.glob("*.npz")):
        participant, route = path.stem.rsplit("_", 1)
        data = np.load(path)
        if len(data["indexes"]) >= 20:
            out.append((participant, route, data["indexes"], data["points"]))
    return out


def show_where():
    print("\nЧТО ПРАВИЛА ВЫДАЮТ НА КАДРАХ (все видео датасета)\n")
    counts = dict.fromkeys(TECHNIQUES, 0)
    counts["ничего не подошло"] = 0
    for _p, _r, _idx, points in cached():
        for row in points:
            name, _ = detect(row)
            counts[name or "ничего не подошло"] += 1
    total = max(1, sum(counts.values()))
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<26}{count:>7}  {100 * count / total:5.1f}%")
    print(f"  {'всего кадров с позой':<26}{total:>7}")


def show_geometry():
    print("\nРАЗБРОС ГЕОМЕТРИИ ПОЗЫ — из него назначаются пороги в technique.py\n")
    rows = {}

    def add(name, value):
        rows.setdefault(name, []).append(value)

    for _p, _r, _idx, points in cached():
        for row in points:
            pose = Pose.of(row)
            add("разворот таза", pose.hip_turn)
            add("сдвиг плеч вбок", pose.shoulder_shift)
            add("стопа вбок (макс)", max(pose.foot_side))
            add("угол колена (макс)", max(pose.knee_angle))
            add("угол колена (мин)", min(pose.knee_angle))
            add("стопы врозь", pose.feet_apart)
            add("разница высот стоп", pose.feet_height_gap)

    print(f"  {'величина':<26}{'p10':>8}{'медиана':>9}{'p90':>8}{'p99':>8}")
    for name, values in rows.items():
        a = np.asarray(values)
        print(f"  {name:<26}{np.percentile(a, 10):8.2f}{np.median(a):9.2f}"
              f"{np.percentile(a, 90):8.2f}{np.percentile(a, 99):8.2f}")


def _crop(image, points):
    height, width = image.shape[:2]
    xs, ys = points[:, 0] * width, points[:, 1] * height
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    pad_x, pad_y = (x1 - x0) * 0.8, (y1 - y0) * 0.25
    box = (
        max(0, int(x0 - pad_x)), min(width, int(x1 + pad_x)),
        max(0, int(y0 - pad_y)), min(height, int(y1 + pad_y)),
    )
    if box[1] - box[0] < 20 or box[3] - box[2] < 20:
        return None
    return image[box[2]:box[3], box[0]:box[1]]


def _grid(images, cols = 6, cell=(240, 340)):
    cells = []
    for image in images:
        canvas = np.full((cell[1], cell[0], 3), 32, np.uint8)
        scale = min(cell[0] / image.shape[1], cell[1] / image.shape[0])
        small = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        top, left = (cell[1] - small.shape[0]) // 2, (cell[0] - small.shape[1]) // 2
        canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
        cells.append(canvas)
    while len(cells) % cols:
        cells.append(np.full((cell[1], cell[0], 3), 32, np.uint8))
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def show_top(technique, out_name, count = 12):
    ranked = []
    for participant, route, indexes, points in cached():
        for frame_index, row in zip(indexes, points, strict=True):
            score = score_all(row).get(technique, 0.0)
            if score > 0:
                ranked.append((float(score), participant, route, int(frame_index)))
    ranked.sort(reverse=True)

    picked, per_video = [], {}
    for score, participant, route, frame_index in ranked:
        if per_video.get((participant, route), 0) >= 3:
            continue
        per_video[(participant, route)] = per_video.get((participant, route), 0) + 1
        picked.append((score, participant, route, frame_index))
        if len(picked) >= count:
            break

    images = []
    for score, participant, route, frame_index in picked:
        capture = cv2.VideoCapture(str(DATASET / participant / f"{route}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        capture.release()
        data = np.load(CACHE / f"{participant}_{route}.npz")
        at = int(np.argmin(np.abs(data["indexes"] - frame_index)))
        piece = _crop(image, data["points"][at]) if ok else None
        if piece is None:
            continue
        piece = piece.copy()
        cv2.rectangle(piece, (0, 0), (piece.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(piece, f"{score:.2f} {participant}/{route}", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(piece)

    if not images:
        print(f"для «{technique}» подходящих кадров не нашлось")
        return
    out = CACHE / f"top_{out_name}.jpg"
    out.write_bytes(cv2.imencode(".jpg", _grid(images))[1].tobytes())
    print(f"{out}  кадров {len(images)}")


def main():
    setup()
    what = sys.argv[1] if len(sys.argv) > 1 else "where"
    if what == "poses":
        build_cache()
    elif what == "where":
        show_where()
    elif what == "geom":
        show_geometry()
    elif what == "top":
        show_top(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "technique")
    else:
        print(СПРАВКА)


if __name__ == "__main__":
    main()
