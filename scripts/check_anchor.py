from collections import defaultdict
from pathlib import Path

import numpy as np

from climb_ai.pose import Frame
from climb_ai.route import find_climb
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
DATASET = Path(r"C:\Users\user\Desktop\база данных")


def load_climb(participant, route):
    path = CACHE / f"{participant}_{route}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    camera = data["camera"] if "camera" in data.files else np.zeros(len(data["indexes"]))
    seen = data["visible"] if "visible" in data.files else [None] * len(data["indexes"])
    return find_climb([
        Frame(index=int(i), second=float(s), points=p, visible=v, camera_y=float(c))
        for i, s, p, c, v in zip(
            data["indexes"], data["seconds"], data["points"], camera, seen, strict=True
        )
    ])


def first_hand_frame(usage_csv):
    out = {}
    for line in usage_csv.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        token = parts[0].strip().lower().replace(" ", "")
        limb = "".join(c for c in token if c.isalpha())[:2]
        digits = "".join(c for c in token if c.isdigit())
        if limb not in ("lh", "rh") or not digits:
            continue
        hold, start = int(digits), int(parts[1])
        out[hold] = min(out.get(hold, start), start)
    return out


def main():
    setup()
    print()
    print("УКАЗЫВАЕТ ЛИ ПРОДВИЖЕНИЕ НА ОДНО И ТО ЖЕ МЕСТО ТРАССЫ")
    print()
    print("Берём момент, когда рука впервые легла на размеченную зацепку,")
    print("и смотрим, насколько разошлось продвижение у разных людей.")
    all_spreads = []

    for route in ("green", "orange"):
        by_hold = defaultdict(list)
        climbers = 0
        for folder in sorted(p for p in DATASET.iterdir() if p.is_dir() and p.name.startswith("p")):
            climb = load_climb(folder.name, route)
            usage = folder / f"{route}_holdUsage.csv"
            if climb is None or not usage.exists():
                continue
            climbers += 1
            indexes = np.array([f.index for f in climb.frames])
            for hold, frame_index in first_hand_frame(usage).items():
                at = int(np.argmin(np.abs(indexes - frame_index)))
                if abs(int(indexes[at]) - frame_index) <= 15:
                    by_hold[hold].append(float(climb.progress[at]))

        print(f"\n  ТРАССА {route.upper()}: пролазов разобрано {climbers}")
        print(f"  {'зацепка':>8}{'продвижение':>14}{'разброс':>10}{'человек':>9}")
        spreads = []
        for hold in sorted(by_hold):
            values = by_hold[hold]
            if len(values) < 4:
                continue
            spread = float(np.std(values))
            spreads.append(spread)
            print(f"  {hold:>8}{np.mean(values):>14.2f}{spread:>10.2f}{len(values):>9}")
        if spreads:
            all_spreads += spreads
            print(f"  --> средний разброс: {np.mean(spreads):.3f}")

    if all_spreads:
        mean = float(np.mean(all_spreads))
        print(f"\n  ИТОГ: средний разброс продвижения на одной зацепке {mean:.3f}")
        print("  Для сравнения: если бы привязка не работала совсем, разброс был")
        print("  бы около 0.29 — столько даёт равномерно случайное продвижение.")
        print(f"  Вывод: привязка {'работает' if mean < 0.15 else 'НЕ работает'}.")


if __name__ == "__main__":
    main()
