from pathlib import Path

from climb_ai.analyze import MATCH_SLACK_SEC
from climb_ai.holds import find_contacts, situations
from climb_ai.pose import frames_from_cache
from climb_ai.reference import Reference, collect_examples
from climb_ai.route import find_climb
from climb_ai.technique import TECHNIQUES, performed
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
HELD_OUT = ("p9", "p10")


def climb_of(path):
    return find_climb(frames_from_cache(path))


def build(route):
    reference = Reference()
    for path in sorted(CACHE.glob(f"*_{route}.npz")):
        who = path.stem.rsplit("_", 1)[0]
        if who in HELD_OUT:
            continue
        climb = climb_of(path)
        if climb is None:
            continue
        examples = collect_examples(climb, f"{who}/{route}")
        if examples:
            reference.examples.extend(examples)
            reference.climbs.append(f"{who}/{route}")
    return reference


def held_out_climbs():
    for route in ("green", "orange"):
        reference = build(route)
        if reference.total < 2:
            continue
        for who in HELD_OUT:
            path = CACHE / f"{who}_{route}.npz"
            if not path.exists():
                continue
            climb = climb_of(path)
            if climb is not None:
                yield f"{who}/{route}", reference, climb


def background(done, seconds):
    length = max(1e-6, seconds[1] - seconds[0])
    out = {}
    for name in TECHNIQUES:
        held = sum(m.seconds for m in done if m.technique == name)
        out[name] = min(1.0, held / length)
    return out


def main():
    setup()
    print("\nСБЫВАЮТСЯ ЛИ ТРЕБОВАНИЯ ЭТАЛОНА НА ОТЛОЖЕННЫХ УЧАСТНИКАХ\n")
    print(f"  {'видео':<14}{'требований':>12}{'выполнено':>11}{'сбылось':>9}{'фон':>7}{'польза':>8}")

    всего = сбылось = 0
    фоны = []
    for имя, reference, climb in held_out_climbs():
        contacts = find_contacts(climb.frames)
        мои = situations(climb.frames, contacts, climb.progress)
        сделал = performed(climb.frames, climb.progress, contacts)
        фон = background(сделал, climb.seconds)

        требований = попало = 0
        сумма_фона = 0.0
        for situation in мои:
            for name, _share in reference.required_at(situation):
                требований += 1
                сумма_фона += фон.get(name, 0.0)
                попало += any(
                    m.technique == name
                    and m.start_sec - MATCH_SLACK_SEC <= situation.second <= m.end_sec + MATCH_SLACK_SEC
                    for m in сделал
                )
        if not требований:
            print(f"  {имя:<14}{0:>12}{len(сделал):>11}{'—':>9}{'—':>7}{'—':>8}")
            continue

        доля = попало / требований
        средний_фон = сумма_фона / требований
        всего += требований
        сбылось += попало
        фоны.append(средний_фон * требований)
        print(f"  {имя:<14}{требований:>12}{len(сделал):>11}{доля:>9.0%}"
              f"{средний_фон:>7.0%}{доля - средний_фон:>+8.0%}")

    if всего:
        общий_фон = sum(фоны) / всего
        print(f"\n  ИТОГО: сбылось {сбылось}/{всего} = {сбылось / всего:.0%}, "
              f"фон {общий_фон:.0%}, польза {сбылось / всего - общий_фон:+.0%}")
        print("  Польза — это то, насколько требование точнее простого «он и так")
        print("  часто это делает». Ноль или минус означают, что требовать нечего.")


if __name__ == "__main__":
    main()
