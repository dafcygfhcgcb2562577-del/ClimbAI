import sys
from collections import Counter
from pathlib import Path

import numpy as np

from climb_ai.holds import find_contacts, hold_map, situations
from climb_ai.pose import frames_from_cache
from climb_ai.route import find_climb
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
DATASET = Path(r"C:\Users\user\Desktop\база данных")
FPS = 25.0

LIMB_CODE = {"левая рука": "lh", "правая рука": "rh", "левая нога": "lf", "правая нога": "rf"}


def load_climb(path):
    return find_climb(frames_from_cache(path))


def annotated_grips(participant, route):
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
        if limb not in ("lh", "rh", "lf", "rf") or not digits:
            continue
        try:
            out.append({"limb": limb, "hold": int(digits),
                        "from": int(parts[1]) / FPS, "to": int(parts[2]) / FPS})
        except ValueError:
            continue
    return out


def check_contacts():
    print("\nНАЙДЕННЫЕ ХВАТЫ ПРОТИВ РУЧНОЙ РАЗМЕТКИ\n")
    print(f"  {'видео':<14}{'найдено':>9}{'размечено':>11}{'совпало':>9}{'доля':>8}")
    всего_найдено = всего_совпало = всего_размечено = 0

    for path in sorted(CACHE.glob("*.npz")):
        participant, route = path.stem.rsplit("_", 1)
        climb = load_climb(path)
        grips = annotated_grips(participant, route)
        if climb is None or not grips:
            continue

        contacts = find_contacts(climb.frames)
        внутри = [g for g in grips if g["to"] >= climb.frames[0].second
                  and g["from"] <= climb.frames[-1].second]
        совпало = sum(
            1 for c in contacts
            if any(LIMB_CODE[c.limb] == g["limb"] and c.to_sec >= g["from"] and c.from_sec <= g["to"]
                   for g in внутри)
        )
        доля = совпало / max(1, len(contacts))
        print(f"  {participant + '/' + route:<14}{len(contacts):>9}{len(внутри):>11}"
              f"{совпало:>9}{доля:>8.2f}")
        всего_найдено += len(contacts)
        всего_совпало += совпало
        всего_размечено += len(внутри)

    print(f"\n  Из найденных хватов попали в размеченный вручную: "
          f"{всего_совпало}/{всего_найдено} = {всего_совпало / max(1, всего_найдено):.2f}")
    print("  Если бы хваты находились наугад, доля была бы заметно ниже:")
    print("  разметка покрывает не всё время пролаза.")


def check_similarity():
    print("\nПОХОЖАЯ ГЕОМЕТРИЯ ЗАЦЕПОК — ЭТО ОДНО МЕСТО ТРАССЫ?\n")

    for route in ("green", "orange"):
        собрано = []
        for path in sorted(CACHE.glob(f"*_{route}.npz")):
            participant = path.stem.rsplit("_", 1)[0]
            climb = load_climb(path)
            grips = annotated_grips(participant, route)
            if climb is None or not grips:
                continue
            contacts = find_contacts(climb.frames)
            for s in situations(climb.frames, contacts, climb.progress):
                руки = sorted(
                    g["hold"] for g in grips
                    if g["limb"] in ("lh", "rh") and g["from"] <= s.second <= g["to"]
                )
                if руки:
                    собрано.append((participant, s, tuple(руки)))
        if len(собрано) < 50:
            print(f"  {route}: слишком мало раскладов ({len(собрано)})")
            continue

        совпало = попыток = 0
        случайно = 0
        for участник, расклад, зацепки in собрано:
            чужие = [(p, s, h) for p, s, h in собрано if p != участник]
            if not чужие:
                continue
            ближайший = min(чужие, key=lambda item: расклад.distance_to(item[1]))
            попыток += 1
            if set(зацепки) & set(ближайший[2]):
                совпало += 1
            наугад = чужие[np.random.default_rng(попыток).integers(len(чужие))]
            if set(зацепки) & set(наугад[2]):
                случайно += 1

        print(f"  {route}: раскладов {len(собрано)}, сравнений {попыток}")
        print(f"    ближайший по геометрии попал в ту же зацепку: {совпало / max(1, попыток):.2f}")
        print(f"    случайный расклад попал бы:                    {случайно / max(1, попыток):.2f}")


def check_map():
    print("\nКАРТА ЗАЦЕПОК\n")
    print(f"  {'видео':<14}{'найдено':>9}{'в разметке':>12}")
    for path in sorted(CACHE.glob("*.npz")):
        participant, route = path.stem.rsplit("_", 1)
        climb = load_climb(path)
        grips = annotated_grips(participant, route)
        if climb is None or not grips:
            continue
        карта = hold_map(find_contacts(climb.frames))
        использовано = len({g["hold"] for g in grips})
        print(f"  {participant + '/' + route:<14}{len(карта):>9}{использовано:>12}")


def check_technique():
    print("\nПОХОЖИЕ ЗАЦЕПКИ — ПОХОЖАЯ ТЕХНИКА?\n")
    from climb_ai.technique import smoothed_scores, techniques_of

    собрано = []
    for path in sorted(CACHE.glob("*.npz")):
        participant, route = path.stem.rsplit("_", 1)
        climb = load_climb(path)
        if climb is None:
            continue
        contacts = find_contacts(climb.frames)
        строки = smoothed_scores(climb.frames)
        наборы = {int(f.index): techniques_of(r)
                  for f, r in zip(climb.frames, строки, strict=True)}
        for s in situations(climb.frames, contacts, climb.progress):
            собрано.append((f"{participant}/{route}", s, наборы.get(s.frame_index, frozenset())))

    print(f"  раскладов собрано: {len(собрано)}")
    есть_техника = [x for x in собрано if x[2]]
    print(f"  из них с распознанной техникой: {len(есть_техника)}")
    if len(есть_техника) < 50:
        print("  слишком мало, чтобы делать выводы")
        return

    совпало = случайно = попыток = 0
    rng = np.random.default_rng(1)
    for видео, расклад, техника in есть_техника:
        чужие = [x for x in есть_техника if x[0] != видео]
        if not чужие:
            continue
        попыток += 1
        ближайший = min(чужие, key=lambda item: расклад.distance_to(item[1]))
        совпало += int(bool(ближайший[2] & техника))
        случайно += int(bool(чужие[rng.integers(len(чужие))][2] & техника))

    print("\n  У ДРУГОГО человека на ближайшем по геометрии раскладе")
    print(f"    техника та же:                 {совпало / max(1, попыток):.2f}")
    print(f"    на случайном раскладе была бы: {случайно / max(1, попыток):.2f}")
    от_частой = Counter(имя for _, _, набор in есть_техника for имя in набор)
    самая_частая = от_частой.most_common(1)[0][1] / len(есть_техника)
    print(f"    если всегда называть самую частую: {самая_частая:.2f}")


if __name__ == "__main__":
    setup()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "technique"):
        check_technique()
    if what in ("all", "contacts"):
        check_contacts()
    if what in ("all", "map"):
        check_map()
    if what in ("all", "similar"):
        check_similarity()
