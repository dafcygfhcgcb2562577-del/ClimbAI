import argparse
import json
from pathlib import Path

from climb_ai.analyze import _find_spots
from climb_ai.holds import find_contacts, situations
from climb_ai.pose import frames_from_cache
from climb_ai.reference import Reference, collect_examples
from climb_ai.route import find_climb
from climb_ai.technique import performed
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
SAVED = PROJECT_ROOT / "artifacts" / "_benchmark"

HELD_OUT = ("p9", "p10")


def load_climb(participant, route):
    path = CACHE / f"{participant}_{route}.npz"
    if not path.exists():
        return None
    return find_climb(frames_from_cache(path))


def participants(route):
    return sorted(path.stem.rsplit("_", 1)[0] for path in CACHE.glob(f"*_{route}.npz"))


def build_reference(route):
    reference = Reference()
    for who in participants(route):
        if who in HELD_OUT:
            continue
        climb = load_climb(who, route)
        if climb is None:
            continue
        examples = collect_examples(climb, f"{who}/{route}")
        if examples:
            reference.examples.extend(examples)
            reference.climbs.append(f"{who}/{route}")
    return reference


def measure():
    rows = []
    for route in ("green", "orange"):
        reference = build_reference(route)
        if reference.total < 2:
            continue
        выигрыш = reference.agreement() - reference.by_chance()
        print(f"  {route}: эталон из {reference.total} пролазов, "
              f"моментов {len(reference.examples)}, "
              f"согласие {reference.agreement():.0%}, "
              f"случайное {reference.by_chance():.0%}, "
              f"выигрыш {выигрыш:+.0%}")

        for who in HELD_OUT:
            climb = load_climb(who, route)
            if climb is None:
                continue
            contacts = find_contacts(climb.frames)
            done = performed(climb.frames, climb.progress, contacts)
            spots = _find_spots(
                situations(climb.frames, contacts, climb.progress),
                done,
                reference,
                climb.frames,
                contacts,
            )
            mistakes = [s for s in spots if not s.done]
            rows.append({
                "видео": f"{who}/{route}",
                "кадров": len(climb),
                "хватов": len(contacts),
                "выполнено": len(done),
                "мест": len(spots),
                "ошибок": len(mistakes),
                "согласие": round(reference.agreement(), 2),
                "техники": sorted({имя for s in mistakes for имя in s.techniques}),
            })
    return rows


COLUMNS = (("кадров", 8), ("хватов", 8), ("выполнено", 11), ("мест", 7), ("ошибок", 8))


def show(rows):
    print(f"\n  {'видео':<14}" + "".join(f"{name:>{width}}" for name, width in COLUMNS)
          + "   что не сделано")
    print("  " + "-" * 78)
    for row in rows:
        print(f"  {row['видео']:<14}"
              + "".join(f"{row[name]:>{width}}" for name, width in COLUMNS)
              + "   " + (", ".join(row["техники"]) if row["техники"] else "—"))
    print("  " + "-" * 78)
    print(f"  {'итого':<14}"
          + "".join(f"{sum(r[name] for r in rows):>{width}}" for name, width in COLUMNS))
    с_ошибками = sum(1 for r in rows if r["ошибок"])
    print(f"\n  ошибки найдены на {с_ошибками} видео из {len(rows)}")


def diff(before, after):
    было = {r["видео"]: r for r in before}
    print("\n  СРАВНЕНИЕ (было -> стало)")
    for row in after:
        old = было.get(row["видео"], {})
        parts = []
        for name, _ in COLUMNS:
            a, b = old.get(name), row[name]
            parts.append(f"{name} {a}->{b}" + ("" if a == b else (" +" if (b or 0) > (a or 0) else " -")))
        print(f"  {row['видео']:<14}" + ", ".join(parts))


def main():
    setup()
    parser = argparse.ArgumentParser(description="Замер разбора ошибок")
    parser.add_argument("--save", metavar="ИМЯ", help="сохранить замер под этим именем")
    parser.add_argument("--diff", metavar="ИМЯ", help="сравнить с сохранённым замером")
    args = parser.parse_args()

    if not CACHE.exists():
        print("Нет кэша поз. Сначала: python -m scripts.audit_technique poses")
        return

    print("Замер: эталон по одним участникам, проверка по другим.\n")
    rows = measure()
    if not rows:
        print("Нечего мерить: в кэше нет подходящих видео")
        return
    show(rows)

    SAVED.mkdir(parents=True, exist_ok=True)
    if args.diff:
        path = SAVED / f"{args.diff}.json"
        if path.exists():
            diff(json.loads(path.read_text(encoding="utf-8")), rows)
        else:
            print(f"\n  нет сохранённого замера «{args.diff}»")
    if args.save:
        path = SAVED / f"{args.save}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  замер сохранён: {path}")


if __name__ == "__main__":
    main()
