from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from climb_ai.pose import frames_from_cache
from climb_ai.route import find_climb
from climb_ai.technique import SMOOTH_SEC, smoothed_scores, techniques_of
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
DATASET = Path(r"C:\Users\user\Desktop\база данных")
FPS = 25.0


def hand_grips(participant, route):
    path = DATASET / participant / f"{route}_holdUsage.csv"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        token = parts[0].lower().replace(" ", "")
        limb = "".join(c for c in token if c.isalpha())[:2]
        digits = "".join(c for c in token if c.isdigit())
        if limb not in ("lh", "rh") or not digits:
            continue
        try:
            out.append({"limb": limb, "hold": int(digits),
                        "from": int(parts[1]) / FPS, "to": int(parts[2]) / FPS})
        except ValueError:
            continue
    return out


def videos():
    for path in sorted(CACHE.glob("*.npz")):
        participant, route = path.stem.rsplit("_", 1)
        grips = hand_grips(participant, route)
        if not grips:
            continue
        frames = frames_from_cache(path)
        climb = find_climb(frames)
        if climb is None:
            continue
        yield participant, route, climb, grips


def techniques_by_second(frames):
    строки = smoothed_scores(frames, SMOOTH_SEC)
    return {float(f.second): techniques_of(r) for f, r in zip(frames, строки, strict=True)}


ПРОБ = 300


def согласие_в(группы):
    голоса = 0
    совпало = 0
    for мнения in группы:
        счёт = Counter(имя for набор in мнения for имя in набор)
        победитель, сколько = счёт.most_common(1)[0]
        голоса += len(мнения)
        совпало += сколько
    return совпало / max(1, голоса)


def случайный_уровень(группы):
    if not группы:
        return 0.0, 0.0
    все = [мнение for мнения in группы for мнение in мнения]
    кости = np.random.default_rng(1)
    доли = []
    for _ in range(ПРОБ):
        порядок = кости.permutation(len(все))
        мешок = [все[номер] for номер in порядок]
        взято = 0
        перемешанные = []
        for мнения in группы:
            перемешанные.append(мешок[взято : взято + len(мнения)])
            взято += len(мнения)
        доли.append(согласие_в(перемешанные))
    return float(np.mean(доли)), float(np.std(доли))


def nearest_second(по_секундам, second):
    if not по_секундам:
        return frozenset()
    ближайшая = min(по_секундам, key=lambda s: abs(s - second))
    return по_секундам[ближайшая] if abs(ближайшая - second) <= 1.0 else frozenset()


def main():
    setup()
    print("\nСОГЛАСНЫ ЛИ УЧАСТНИКИ НА ОДНОЙ И ТОЙ ЖЕ РАЗМЕЧЕННОЙ ЗАЦЕПКЕ\n")
    print("Место берём из ручной разметки, а не из геометрии: тут ошибиться нельзя.\n")

    по_зацепке = defaultdict(dict)
    всего_хватов = с_техникой = 0

    for participant, route, climb, grips in videos():
        по_секундам = techniques_by_second(climb.frames)
        внутри = [
            g for g in grips
            if g["to"] >= climb.frames[0].second and g["from"] <= climb.frames[-1].second
        ]
        for g in внутри:
            середина = 0.5 * (g["from"] + g["to"])
            техники = nearest_second(по_секундам, середина)
            всего_хватов += 1
            с_техникой += bool(техники)
            прежние = по_зацепке[(route, g["hold"])].get(participant, frozenset())
            по_зацепке[(route, g["hold"])][participant] = прежние | техники

    print(f"  размеченных хватов рукой внутри пролаза: {всего_хватов}")
    print(f"  из них с распознанной техникой: {с_техникой} "
          f"({100 * с_техникой / max(1, всего_хватов):.0f}%)\n")

    группы = []
    строки = []
    for (route, hold), по_людям in sorted(по_зацепке.items()):
        мнения = [t for t in по_людям.values() if t]
        if len(мнения) < 3:
            continue
        счёт = Counter(имя for набор in мнения for имя in набор)
        победитель, сколько = счёт.most_common(1)[0]
        группы.append(мнения)
        строки.append((route, hold, len(по_людям), len(мнения), победитель, сколько / len(мнения)))

    print(f"  {'трасса':<8}{'зацепка':>8}{'людей':>7}{'с техникой':>12}"
          f"{'что делают':>24}{'согласны':>10}")
    for route, hold, людей, мнений, победитель, доля in строки:
        print(f"  {route:<8}{hold:>8}{людей:>7}{мнений:>12}{победитель:>24}{доля:>10.0%}")

    наблюдаемое = согласие_в(группы)
    случайное, разброс = случайный_уровень(группы)
    print()
    print(f"  СОГЛАСИЕ на одной и той же зацепке: {наблюдаемое:.0%}"
          f"  (зацепок с тремя и более мнениями: {len(строки)})")
    print(f"  Случайный уровень на тех же метках: {случайное:.0%} "
          f"плюс-минус {разброс:.0%}")
    print(f"  Выигрыш над случайным: {наблюдаемое - случайное:+.0%}")


if __name__ == "__main__":
    main()
