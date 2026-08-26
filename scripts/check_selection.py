from pathlib import Path

from climb_ai.analyze import MATCH_SLACK_SEC, _find_spots
from climb_ai.holds import find_contacts, situations
from climb_ai.pose import frames_from_cache
from climb_ai.reference import (
    Reference,
    collect_examples,
    drop_useless,
    gain_of,
    sources_of,
)
from climb_ai.route import find_climb
from climb_ai.technique import performed
from scripts._console import setup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE = PROJECT_ROOT / "artifacts" / "_poses"
HELD_OUT = ("p9", "p10")


def climb_of(path):
    return find_climb(frames_from_cache(path))


def learning_pool():
    reference = Reference()
    for path in sorted(CACHE.glob("*.npz")):
        who = path.stem.rsplit("_", 1)[0]
        if who in HELD_OUT:
            continue
        climb = climb_of(path)
        if climb is None:
            continue
        examples = collect_examples(climb, path.stem)
        if examples:
            reference.examples.extend(examples)
            reference.climbs.append(path.stem)
    return reference


def held_out_climbs():
    for path in sorted(CACHE.glob("*.npz")):
        who = path.stem.rsplit("_", 1)[0]
        if who not in HELD_OUT:
            continue
        climb = climb_of(path)
        if climb is not None:
            yield path.stem, climb


def score_on_held_out(reference):
    требований = 0
    сбылось = 0
    мест = 0
    ошибок = 0
    for имя, climb in held_out_climbs():
        contacts = find_contacts(climb.frames)
        мои = situations(climb.frames, contacts, climb.progress)
        сделал = performed(climb.frames, climb.progress, contacts)
        for situation in мои:
            for name, доля in reference.required_at(situation):
                требований += 1
                попало = any(
                    m.technique == name
                    and m.start_sec - MATCH_SLACK_SEC <= situation.second
                    and situation.second <= m.end_sec + MATCH_SLACK_SEC
                    for m in сделал
                )
                сбылось += int(попало)
        места = _find_spots(мои, сделал, reference, climb.frames, contacts)
        мест += len(места)
        ошибок += sum(1 for место in места if not место.done)
    доля = сбылось / требований if требований else 0.0
    return требований, доля, мест, ошибок


def показать(подпись, reference):
    требований, доля, мест, ошибок = score_on_held_out(reference)
    print(f"  {подпись:<22}{len(sources_of(reference)):>9}"
          f"{gain_of(reference):>+10.1%}{требований:>13}{доля:>10.0%}{мест:>7}{ошибок:>8}")


def main():
    setup()
    print()
    print("ОТБОР ПРОЛАЗОВ В ЭТАЛОН: помогает ли он на отложенных участниках")
    print()
    print("Эталон учится на всех, кроме p9 и p10, проверяется на них.")
    print()
    print(f"  {'эталон':<22}{'пролазов':>9}{'выигрыш':>10}"
          f"{'требований':>13}{'сбылось':>10}{'мест':>7}{'ошибок':>8}")

    все = learning_pool()
    показать("все пролазы", все)

    убранные = []
    отобранный = drop_useless(все, lambda имя, было, стало: убранные.append(имя))
    показать("после отбора", отобранный)

    print()
    if убранные:
        print(f"  убрано пролазов: {len(убранные)}")
        for имя in убранные:
            print(f"    {имя}")
    else:
        print("  отбор ничего не убрал")


if __name__ == "__main__":
    main()
