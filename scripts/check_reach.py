import sys
from pathlib import Path

import numpy as np

from climb_ai.holds import HANDS, LIMBS, find_contacts, situations, wall_points
from climb_ai.pose import frames_from_cache
from climb_ai.route import find_climb
from climb_ai.technique import Pose, score_all
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"


def load_climb(path):
    return find_climb(frames_from_cache(path))


def videos():
    for path in sorted(CACHE.glob("*.npz")):
        climb = load_climb(path)
        if climb is not None and len(climb) >= 20:
            yield path.stem, climb


def check_labels_stable():
    print("\n1. УСТОЙЧИВЫ ЛИ САМИ МЕТКИ ТЕХНИК\n")
    свои = чужие = свои_всего = чужие_всего = 0

    все = []
    for имя, climb in videos():
        contacts = find_contacts(climb.frames)
        по_кадру = {int(f.index): f.points for f in climb.frames}
        for s in situations(climb.frames, contacts, climb.progress):
            points = по_кадру.get(s.frame_index)
            if points is None:
                continue
            scores = score_all(points)
            best = max(scores.items(), key=lambda kv: kv[1])
            если = best[0] if best[1] >= 0.35 else None
            все.append((имя, s, если))

    с_техникой = [x for x in все if x[2] is not None]
    for видео, расклад, техника in с_техникой:
        внутри = [x for x in с_техникой if x[0] == видео and x[1].second != расклад.second]
        снаружи = [x for x in с_техникой if x[0] != видео]
        if внутри:
            свои += int(min(внутри, key=lambda i: расклад.distance_to(i[1]))[2] == техника)
            свои_всего += 1
        if снаружи:
            чужие += int(min(снаружи, key=lambda i: расклад.distance_to(i[1]))[2] == техника)
            чужие_всего += 1

    print(f"  ближайший расклад В ТОМ ЖЕ видео даёт ту же технику: "
          f"{свои / max(1, свои_всего):.2f}")
    print(f"  ближайший расклад В ДРУГОМ видео даёт ту же технику: "
          f"{чужие / max(1, чужие_всего):.2f}")
    print("\n  Если первое высоко, а второе нет — метки в порядке, не переносится привязка.")
    print("  Если оба низкие — ненадёжны сами метки техник.")


def reaches(climb, contacts):
    points = wall_points(climb.frames)
    seconds = np.array([f.second for f in climb.frames])
    out = []
    for hand in HANDS:
        свои = sorted(
            (c for c in contacts if c.limb == hand), key=lambda c: c.from_sec
        )
        column = LIMBS.index(hand)
        for прошлый, следующий in zip(свои, свои[1:], strict=False):
            размах = float(np.hypot(следующий.x - прошлый.x, следующий.y - прошлый.y))
            маска = (seconds >= прошлый.to_sec) & (seconds <= следующий.from_sec)
            if размах < 0.2 or not maska_ok(маска):
                continue
            out.append({
                "hand": hand,
                "размах": размах,
                "кадры": np.flatnonzero(маска),
                "от": float(прошлый.to_sec),
                "до": float(следующий.from_sec),
                "column": column,
                "points": points,
            })
    return out


def maska_ok(маска):
    return bool(np.any(маска))


def check_reach_vs_turn():
    print("\n2. РАЗМАХ ХОДА ПРОТИВ РАЗВОРОТА ТАЗА\n")
    размахи, развороты = [], []

    for _, climb in videos():
        contacts = find_contacts(climb.frames)
        for ход in reaches(climb, contacts):
            куски = [climb.frames[i] for i in ход["кадры"]]
            if not куски:
                continue
            поворот = min(Pose.of(f.points).hip_turn for f in куски)
            размахи.append(ход["размах"])
            развороты.append(поворот)

    if len(размахи) < 30:
        print(f"  ходов найдено всего {len(размахи)} — мало для выводов")
        return

    размахи = np.array(размахи)
    развороты = np.array(развороты)
    связь = float(np.corrcoef(размахи, развороты)[0, 1])
    print(f"  ходов разобрано: {len(размахи)}")
    print(f"  связь размаха и разворота таза: {связь:+.2f}")
    print("  (отрицательная означает: чем дальше тянешься, тем сильнее разворот)\n")

    порог = float(np.median(размахи))
    близкие = развороты[размахи <= порог]
    далёкие = развороты[размахи > порог]
    print(f"  {'ходы':<24}{'сколько':>9}{'разворот таза':>16}")
    print(f"  {'короткие (<= медианы)':<24}{len(близкие):>9}{np.median(близкие):>16.2f}")
    print(f"  {'длинные (> медианы)':<24}{len(далёкие):>9}{np.median(далёкие):>16.2f}")
    print(f"\n  размах: медиана {np.median(размахи):.2f}, p90 {np.percentile(размахи, 90):.2f} "
          f"длин торса")


if __name__ == "__main__":
    setup()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "labels"):
        check_labels_stable()
    if what in ("all", "reach"):
        check_reach_vs_turn()
