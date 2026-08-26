from dataclasses import asdict, dataclass, field

from climb_ai.holds import LEGS, find_contacts, situations
from climb_ai.pose import pose_change, read_poses
from climb_ai.route import find_climb
from climb_ai.technique import Moment, Technique, can_be_seen, performed

MATCH_SLACK_SEC = 1.5

SAME_SPOT_SEC = 2.5

SAME_POSE = 0.48

CLEAN_SHARE = 0.7

KNOWN_ENOUGH = 0.5

MIN_SEEN = 0.5

FLAG_LEG = {Technique.FLAG_LEFT: LEGS[0], Technique.FLAG_RIGHT: LEGS[1]}

MAX_SHOWN = 5


@dataclass
class Spot:
    at_sec: float
    frame_index: int
    techniques: list[str]
    done: bool
    share: float = 0.0
    image: str | None = None

    @property
    def title(self):
        return ", ".join(self.techniques)


@dataclass
class Report:
    video: str
    climb_sec: tuple[float, float]
    done: list[Moment] = field(default_factory=list)
    spots: list[Spot] = field(default_factory=list)
    note: str = ""
    checked: bool = False
    confidence: float = 0.0
    known_share: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def mistakes(self):
        return [spot for spot in self.spots if not spot.done]

    @property
    def correct(self):
        return [spot for spot in self.spots if spot.done]

    def as_dict(self):
        return {
            "video": self.video,
            "climb_sec": list(self.climb_sec),
            "done": [asdict(moment) for moment in self.done],
            "mistakes": [asdict(spot) for spot in self.mistakes],
            "correct": [asdict(spot) for spot in self.correct],
            "note": self.note,
            "checked": self.checked,
            "confidence": round(self.confidence, 2),
            "known_share": round(self.known_share, 2),
            "stats": self.stats,
        }


def analyze(
    video,
    engine,
    reference = None,
    sample_step = 5,
    on_progress = None,
):
    if on_progress:
        on_progress("Разбираю видео", 0.05)
    frames = read_poses(video, engine, sample_step, lambda part: _tick(on_progress, part))

    if on_progress:
        on_progress("Определяю границы пролаза", 0.80)
    climb = find_climb(frames)
    if climb is None:
        return Report(
            video=str(video),
            climb_sec=(0.0, 0.0),
            note=_why_no_climb(frames),
            stats=_stats(frames, None, []),
        )

    if on_progress:
        on_progress("Разбираю технику", 0.90)
    contacts = find_contacts(climb.frames)
    mine = situations(climb.frames, contacts, climb.progress)
    report = Report(
        video=str(video),
        climb_sec=climb.seconds,
        done=performed(climb.frames, climb.progress, contacts),
        stats=_stats(frames, climb, contacts),
    )

    if reference is None:
        report.note = "Базы эталонов нет, поэтому показано только выполненное."
        return report

    report.checked = True
    report.confidence = reference.agreement()
    report.spots = _hide_nitpicks(
        _find_spots(mine, report.done, reference, climb.frames, contacts)
    )
    report.known_share = _known_share(mine, reference)

    if not mine:
        report.note = (
            "Не видно, за какие зацепки человек держался. Обычно так выходит, "
            "когда камера идёт вплотную за спортсменом: стены в кадре почти "
            "нет, и понять, что неподвижно, не по чему. Снимайте с одной точки, "
            "чтобы человек и стена помещались целиком."
        )
    elif not report.spots:
        report.note = (
            "На этих зацепках эталонные пролазы обходятся без техники."
            if report.known_share >= KNOWN_ENOUGH
            else "Таких зацепок в эталонах нет, сравнить не с чем."
        )
    return report


def _known_share(mine, reference):
    if not mine:
        return 0.0
    return sum(1 for situation in mine if reference.knows_place(situation)) / len(mine)


def _find_spots(mine, done, reference, frames=None, contacts=None):
    frames = list(frames or [])
    by_frame = {int(frame.index): frame for frame in frames}
    found = []
    for situation in mine:
        required = _one_flag(reference.required_at(situation))
        if not required:
            continue
        chosen, share = _main_demand(required, done, situation.second)
        сделан = _moment_of(done, chosen, situation.second)
        if сделан is not None:
            промах = (сделан.peak_sec, сделан.peak_index)
        else:
            промах = _missed_moment(chosen, situation, frames, contacts)
            if промах is None:
                continue
        когда, кадр = промах
        found.append(
            Spot(
                at_sec=round(когда, 1),
                frame_index=кадр,
                techniques=[chosen],
                done=сделан is not None,
                share=share,
            )
        )
    return _best_of(_thin_out(found, by_frame))


def _missed_moment(technique, situation, frames, contacts):
    if not frames:
        return situation.second, situation.frame_index
    годные = [
        frame for frame in frames
        if abs(frame.second - situation.second) <= MATCH_SLACK_SEC
        and can_be_seen(frame.points, frame.visible, frame.aspect)[technique] >= MIN_SEEN
        and _leg_free_at(technique, frame.second, contacts)
    ]
    if not годные:
        return None
    ближайший = min(годные, key=lambda frame: abs(frame.second - situation.second))
    return ближайший.second, ближайший.index


def _leg_free_at(technique, second, contacts):
    leg = FLAG_LEG.get(technique)
    if leg is None or not contacts:
        return True
    return not any(
        contact.limb == leg and contact.from_sec <= second <= contact.to_sec
        for contact in contacts
    )


def _best_of(spots):
    ошибки = sorted((s for s in spots if not s.done), key=lambda s: -s.share)[:MAX_SHOWN]
    сделано = sorted((s for s in spots if s.done), key=lambda s: -s.share)[:MAX_SHOWN]
    return sorted(ошибки + сделано, key=lambda s: s.at_sec)


def _main_demand(
    required, done, second
):
    for name, share in required:
        if not _did_it(done, name, second):
            return name, share
    return required[0]


def _hide_nitpicks(spots):
    if not spots:
        return spots
    done = [spot for spot in spots if spot.done]
    if len(done) / len(spots) < CLEAN_SHARE:
        return spots
    return done


def _one_flag(required):
    флажки = [pair for pair in required if pair[0] in (Technique.FLAG_LEFT, Technique.FLAG_RIGHT)]
    if len(флажки) < 2:
        return required
    слабее = min(флажки, key=lambda pair: pair[1])[0]
    return [pair for pair in required if pair[0] != слабее]


def _did_it(done, technique, second):
    return _moment_of(done, technique, second) is not None


def _moment_of(done, technique, second):
    рядом = [
        moment for moment in done
        if moment.technique == technique
        and moment.start_sec - MATCH_SLACK_SEC <= second <= moment.end_sec + MATCH_SLACK_SEC
    ]
    if not рядом:
        return None
    return min(рядом, key=lambda moment: abs(moment.peak_sec - second))


def _thin_out(spots, by_frame):
    kept = []
    for spot in sorted(spots, key=lambda s: (s.at_sec, not s.done)):
        if not _same_place(spot, kept, by_frame):
            kept.append(spot)
    return _drop_satisfied(kept)


def _same_place(spot, kept, by_frame):
    for other in kept:
        рядом = abs(other.at_sec - spot.at_sec) < SAME_SPOT_SEC
        if other.techniques != spot.techniques:
            if рядом:
                return True
            continue
        if рядом or _same_pose(spot, other, by_frame):
            return True
    return False


def _same_pose(spot, other, by_frame):
    здесь, там = by_frame.get(spot.frame_index), by_frame.get(other.frame_index)
    if здесь is None or там is None:
        return False
    return pose_change(здесь, там) < SAME_POSE


def _drop_satisfied(spots):
    сделано = [spot for spot in spots if spot.done]
    kept = []
    for spot in spots:
        if spot.done:
            kept.append(spot)
            continue
        осталось = [
            name for name in spot.techniques
            if not any(
                name in other.techniques and abs(other.at_sec - spot.at_sec) < SAME_SPOT_SEC
                for other in сделано
            )
        ]
        if осталось:
            spot.techniques = осталось
            kept.append(spot)
    return kept


def _stats(frames, climb, contacts):
    return {
        "кадров с позой": len(frames),
        "кадров в пролазе": len(climb) if climb else 0,
        "хватов найдено": len(contacts),
    }


def _why_no_climb(frames):
    if len(frames) < 10:
        return (
            f"Человека почти не видно: поза найдена на {len(frames)} кадрах. "
            "Снимайте так, чтобы спортсмен помещался в кадр целиком."
        )
    return (
        "Подъёма по стене не видно: человек не оторвался от земли или "
        "пролаза на видео нет."
    )


def _tick(on_progress, part):
    if on_progress:
        on_progress("Разбираю видео", 0.05 + 0.75 * part)
