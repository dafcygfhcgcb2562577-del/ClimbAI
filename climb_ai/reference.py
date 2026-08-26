import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from climb_ai.holds import Situation, find_contacts, situations
from climb_ai.pose import UserError, frames_from_cache, poses_with_cache, read_poses
from climb_ai.route import find_climb
from climb_ai.settings import settings
from climb_ai.technique import TECHNIQUES, smoothed_scores, techniques_of

MAX_MATCH_DIST = 1.6

NEIGHBOURS = 30

REQUIRED_SHARE = 0.5

MIN_WITH_TECHNIQUE = 3
MIN_SOURCES = 2

MIN_ACTIVE_SHARE = 0.2

MIN_CLIMBS = 2

MIN_GAIN_STEP = 0.01

MAX_PER_CLIMB = 120

NOTHING = "без техники"


@dataclass
class Example:
    situation: Situation
    techniques: frozenset[str]
    source: str

    @property
    def active(self):
        return bool(self.techniques)

    def as_dict(self):
        return {
            "техники": sorted(self.techniques),
            "пролаз": self.source,
            "секунда": round(self.situation.second, 2),
            "кадр": self.situation.frame_index,
            "продвижение": round(self.situation.progress, 3),
            "расклад": {
                "руки": [round(v, 3) for v in self.situation.hand_span],
                "следующая зацепка": [round(v, 3) for v in self.situation.next_hold]
                if self.situation.next_hold else None,
                "зацепка слева": [round(v, 3) for v in self.situation.left_foot]
                if self.situation.left_foot else None,
                "зацепка справа": [round(v, 3) for v in self.situation.right_foot]
                if self.situation.right_foot else None,
            },
        }

    @classmethod
    def from_dict(cls, raw):
        layout = raw.get("расклад") or {}
        left, right = layout.get("зацепка слева"), layout.get("зацепка справа")
        ahead = layout.get("следующая зацепка")
        return cls(
            situation=Situation(
                second=float(raw.get("секунда", 0.0)),
                frame_index=int(raw.get("кадр", 0)),
                hand_span=tuple(layout.get("руки") or (0.0, 0.0)),
                left_foot=tuple(left) if left else None,
                right_foot=tuple(right) if right else None,
                next_hold=tuple(ahead) if ahead else None,
                progress=float(raw.get("продвижение", 0.0)),
            ),
            techniques=frozenset(raw.get("техники") or ()),
            source=str(raw.get("пролаз", "")),
        )


@dataclass
class Reference:
    climbs: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    _vectors: object = field(default=None, repr=False, compare=False)
    _known_parts: object = field(default=None, repr=False, compare=False)
    _agreement: float | None = field(default=None, repr=False, compare=False)
    _by_chance: float | None = field(default=None, repr=False, compare=False)
    _tries: int | None = field(default=None, repr=False, compare=False)

    @property
    def total(self):
        return len(self.climbs)

    def knows_place(self, situation):
        return len(self._nearest(situation)) >= MIN_WITH_TECHNIQUE

    def required_at(self, situation):
        near = self._nearest(situation)
        if len(near) < MIN_WITH_TECHNIQUE:
            return []

        active = [pair for pair in near if pair[0].active]
        if len(active) < MIN_WITH_TECHNIQUE:
            return []
        if len(active) / len(near) < MIN_ACTIVE_SHARE:
            return []

        required = []
        for name in TECHNIQUES:
            supporters = [pair[0] for pair in active if name in pair[0].techniques]
            if len(supporters) / len(active) < REQUIRED_SHARE:
                continue
            if len({example.source for example in supporters}) < MIN_SOURCES:
                continue
            required.append((name, len(supporters) / len(active)))
        return sorted(required, key=lambda item: item[1], reverse=True)

    def agreement(self):
        self._measure_once()
        return self._agreement

    def by_chance(self):
        self._measure_once()
        return self._by_chance

    def tries(self):
        self._measure_once()
        return self._tries

    def _measure_once(self):
        измерено = (
            self._agreement is not None
            and self._by_chance is not None
            and self._tries is not None
        )
        if not измерено:
            self._agreement, self._by_chance, self._tries = self._measure_agreement()

    def _measure_agreement(self):
        active = [e for e in self.examples if e.active]
        if len(active) < 2:
            return 0.0, 0.0, 0

        vectors = np.array([e.situation.as_vector for e in active])
        known = np.array([e.situation.known for e in active])
        sources = np.array([e.source for e in active])

        gaps = np.array([self._gaps_to(e.situation, vectors, known) for e in active])
        gaps[sources[:, None] == sources[None, :]] = np.inf
        gaps[gaps > MAX_MATCH_DIST] = np.inf

        dice = np.random.default_rng(1)
        hits = lucky = tries = 0
        for row, example in enumerate(active):
            nearest = int(np.argmin(gaps[row]))
            if not np.isfinite(gaps[row][nearest]):
                continue
            tries += 1
            hits += int(bool(example.techniques & active[nearest].techniques))
            others = np.flatnonzero(sources != example.source)
            if len(others):
                partner = active[int(dice.choice(others))]
                lucky += int(bool(example.techniques & partner.techniques))
        if not tries:
            return 0.0, 0.0, 0
        return hits / tries, lucky / tries, tries

    def _nearest(self, situation):
        if not self.examples:
            return []
        gaps = self._gaps_to(situation, self._matrix(), self._known())
        order = np.argsort(gaps)[:NEIGHBOURS]
        return [
            (self.examples[int(index)], float(gaps[index]))
            for index in order
            if gaps[index] <= MAX_MATCH_DIST
        ]

    @staticmethod
    def _gaps_to(situation, vectors, known):
        shared = known & situation.known
        squares = (vectors - situation.as_vector) ** 2 * shared
        counted = np.maximum(shared.sum(axis=1), 1)
        return np.sqrt(squares.sum(axis=1) * len(situation.known) / counted)

    def _matrix(self):
        if self._vectors is None:
            self._vectors = np.array([e.situation.as_vector for e in self.examples])
        return self._vectors

    def _known(self):
        if self._known_parts is None:
            self._known_parts = np.array([e.situation.known for e in self.examples])
        return self._known_parts

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "пролазов": self.climbs,
            "согласие эталонов": round(self.agreement(), 3),
            "случайное совпадение": round(self.by_chance(), 3),
            "сравнений": self.tries(),
            "примеров": len(self.examples),
            "техники": {
                **{name: sum(1 for e in self.examples if name in e.techniques) for name in TECHNIQUES},
                NOTHING: sum(1 for e in self.examples if not e.active),
            },
            "примеры": [example.as_dict() for example in self.examples],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        if not path.is_file():
            raise UserError(
                f"Не найдена база эталонных пролазов: {path}\n"
                "Соберите её из образцовых пролазов:\n"
                '  python app.py эталон --видео "папка с видео"'
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        reference = cls(
            climbs=list(payload.get("пролазов") or []),
            examples=[Example.from_dict(raw) for raw in payload.get("примеры") or []],
        )
        saved = payload.get("согласие эталонов")
        chance = payload.get("случайное совпадение")
        tries = payload.get("сравнений")
        числа = isinstance(saved, (int, float)) and isinstance(chance, (int, float))
        if числа and isinstance(tries, int):
            reference._agreement = float(saved)
            reference._by_chance = float(chance)
            reference._tries = int(tries)
        return reference


def reference_path():
    return settings.reference


def has_reference():
    return reference_path().is_file()


def collect_examples(climb, source):
    contacts = find_contacts(climb.frames)
    scores = smoothed_scores(climb.frames, contacts=contacts)
    by_frame = {int(frame.index): row for frame, row in zip(climb.frames, scores, strict=True)}

    found = []
    for situation in situations(climb.frames, contacts, climb.progress):
        row = by_frame.get(situation.frame_index)
        if row is None:
            continue
        found.append(
            Example(situation=situation, techniques=techniques_of(row), source=source)
        )
    return _thin_out(found, MAX_PER_CLIMB)


def build(videos, engine, sample_step = 5, cache = None):
    reference = Reference()
    for video in sorted(videos):
        frames = (
            poses_with_cache(video, engine, sample_step, cache)
            if cache
            else read_poses(video, engine, sample_step)
        )
        climb = find_climb(frames)
        if climb is None:
            print(f"  пропуск: на {video.name} человек по стене не лез")
            continue
        examples = collect_examples(climb, video.stem)
        with_technique = sum(1 for e in examples if e.active)
        if not with_technique:
            print(f"  пропуск: на {video.name} не распознано ни одной техники")
            continue
        reference.examples.extend(examples)
        reference.climbs.append(video.name)
        print(f"  разобран: {video.name}, моментов {len(examples)}, "
              f"из них с техникой {with_technique}")

    if len(reference.climbs) < MIN_CLIMBS:
        raise UserError(
            f"Для эталона нужно минимум {MIN_CLIMBS} пролаза, разобрано "
            f"{len(reference.climbs)}.\n"
            "Сложите в одну папку несколько образцовых пролазов."
        )
    return reference


def build_from_cache(cache):
    reference = Reference()
    for stored in sorted(Path(cache).glob("*.npz")):
        climb = find_climb(frames_from_cache(stored))
        if climb is None:
            print(f"  пропуск: на {stored.stem} человек по стене не лез")
            continue
        examples = collect_examples(climb, stored.stem)
        with_technique = sum(1 for e in examples if e.active)
        if not with_technique:
            print(f"  пропуск: на {stored.stem} не распознано ни одной техники")
            continue
        reference.examples.extend(examples)
        reference.climbs.append(stored.stem)
        print(f"  разобран: {stored.stem}, моментов {len(examples)}, "
              f"из них с техникой {with_technique}")

    if len(reference.climbs) < MIN_CLIMBS:
        raise UserError(
            f"Для эталона нужно минимум {MIN_CLIMBS} пролаза, в кэше нашлось "
            f"{len(reference.climbs)}."
        )
    return reference


def sources_of(reference):
    порядок = []
    for example in reference.examples:
        if example.source not in порядок:
            порядок.append(example.source)
    return порядок


def without(reference, source):
    примеры = [e for e in reference.examples if e.source != source]
    пролазы = [name for name in reference.climbs if Path(name).stem != source]
    return Reference(climbs=пролазы, examples=примеры)


def gain_of(reference):
    return reference.agreement() - reference.by_chance()


def drop_useless(reference, on_drop = None):
    лучший = reference
    while len(sources_of(лучший)) > MIN_CLIMBS:
        было = gain_of(лучший)
        кандидат = None
        стало_у_кандидата = 0.0
        for source in sources_of(лучший):
            проба = without(лучший, source)
            if len(sources_of(проба)) < MIN_CLIMBS:
                continue
            стало = gain_of(проба)
            if кандидат is None or стало > стало_у_кандидата:
                кандидат = проба
                стало_у_кандидата = стало
                убираем = source
        if кандидат is None:
            break
        if стало_у_кандидата - было < MIN_GAIN_STEP:
            break
        if on_drop is not None:
            on_drop(убираем, было, стало_у_кандидата)
        лучший = кандидат
    return лучший


def _thin_out(examples, limit):
    if len(examples) <= limit:
        return examples
    picked = np.linspace(0, len(examples) - 1, limit).astype(int)
    return [examples[index] for index in sorted(set(picked))]
