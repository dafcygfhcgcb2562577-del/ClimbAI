from dataclasses import dataclass
from functools import cached_property

import numpy as np

from climb_ai.pose import Lm, torso_unit

LIMBS = ("левая рука", "правая рука", "левая нога", "правая нога")
LIMB_POINT = {
    "левая рука": Lm.LEFT_WRIST,
    "правая рука": Lm.RIGHT_WRIST,
    "левая нога": Lm.LEFT_ANKLE,
    "правая нога": Lm.RIGHT_ANKLE,
}
HANDS = ("левая рука", "правая рука")
LEGS = ("левая нога", "правая нога")

RESTING_SPEED = 0.9

MIN_CONTACT_SEC = 0.25

SAME_HOLD_DIST = 0.35

PROGRESS_WEIGHT = 4.0

NEXT_WEIGHT = 1.0

MAX_FOOTHOLD_DIST = 3.0


@dataclass(frozen=True)
class Contact:
    limb: str
    x: float
    y: float
    from_sec: float
    to_sec: float

    def holds_at(self, second):
        return self.from_sec <= second <= self.to_sec


@dataclass(frozen=True)
class Situation:
    second: float
    frame_index: int
    hand_span: tuple[float, float]
    left_foot: tuple[float, float] | None
    right_foot: tuple[float, float] | None
    next_hold: tuple[float, float] | None = None
    progress: float = 0.0

    @cached_property
    def as_vector(self):
        far = (0.0, MAX_FOOTHOLD_DIST)
        return np.array(
            [
                *self.hand_span,
                *(self.left_foot or far),
                *(self.right_foot or far),
                *((np.array(self.next_hold) * NEXT_WEIGHT) if self.next_hold else far),
                self.progress * PROGRESS_WEIGHT,
            ],
            dtype=np.float64,
        )

    @cached_property
    def known(self):
        left, right = self.left_foot is not None, self.right_foot is not None
        ahead = self.next_hold is not None
        return np.array([True, True, left, left, right, right, ahead, ahead, True])

    def distance_to(self, other):
        return float(np.linalg.norm(self.as_vector - other.as_vector))


def wall_points(frames):
    unit = torso_unit(frames)
    out = np.zeros((len(frames), len(LIMBS), 2), dtype=np.float64)
    for row, frame in enumerate(frames):
        for column, limb in enumerate(LIMBS):
            x, y = frame.xy(LIMB_POINT[limb])
            out[row, column] = ((x * frame.aspect) / unit, (y - frame.camera_y) / unit)
    return out


def find_contacts(frames):
    if len(frames) < 3:
        return []

    points = wall_points(frames)
    seconds = np.array([frame.second for frame in frames], dtype=np.float64)
    found = []

    for column, limb in enumerate(LIMBS):
        track = points[:, column, :]
        resting = _resting(track, seconds)
        for start, stop in _runs(resting):
            piece = track[start : stop + 1]
            duration = float(seconds[stop] - seconds[start])
            if duration < MIN_CONTACT_SEC:
                continue
            found.append(
                Contact(
                    limb=limb,
                    x=float(np.median(piece[:, 0])),
                    y=float(np.median(piece[:, 1])),
                    from_sec=float(seconds[start]),
                    to_sec=float(seconds[stop]),
                )
            )
    return sorted(found, key=lambda contact: contact.from_sec)


def hold_map(contacts):
    places = []
    for contact in contacts:
        point = (contact.x, contact.y)
        if not any(_near(point, known, SAME_HOLD_DIST) for known in places):
            places.append(point)
    return places


def situations(
    frames, contacts, progress = None
):
    holds = hold_map(contacts)
    if not holds:
        return []

    points = wall_points(frames)
    hands = [LIMBS.index(name) for name in HANDS]
    out = []

    for row, frame in enumerate(frames):
        gripping = [
            contact for contact in contacts
            if contact.limb in HANDS and contact.holds_at(frame.second)
        ]
        if len(gripping) < 2:
            continue

        left, right = points[row, hands[0]], points[row, hands[1]]
        centre = (left + right) / 2.0
        out.append(
            Situation(
                second=float(frame.second),
                frame_index=int(frame.index),
                hand_span=(float(right[0] - left[0]), float(right[1] - left[1])),
                left_foot=_nearest_foothold(holds, centre, side=-1),
                right_foot=_nearest_foothold(holds, centre, side=+1),
                next_hold=_next_hand_hold(contacts, frame.second, centre),
                progress=float(progress[row]) if progress is not None else 0.0,
            )
        )
    return out


def _next_hand_hold(
    contacts, second, centre
):
    впереди = [
        contact for contact in contacts
        if contact.limb in HANDS and contact.from_sec > second
    ]
    if not впереди:
        return None
    цель = min(впереди, key=lambda contact: contact.from_sec)
    return float(цель.x - centre[0]), float(цель.y - centre[1])


def _nearest_foothold(
    holds, centre, side
):
    best = None
    best_distance = np.inf
    for x, y in holds:
        dx, dy = x - centre[0], y - centre[1]
        if dy <= 0.3:
            continue
        if dx * side < 0:
            continue
        distance = float(np.hypot(dx, dy))
        if distance < best_distance:
            best, best_distance = (float(dx), float(dy)), distance
    return best if best_distance <= MAX_FOOTHOLD_DIST else None


def _resting(track, seconds):
    speed = np.full(len(track), np.inf)
    for index in range(1, len(track)):
        step = float(seconds[index] - seconds[index - 1])
        if step > 0:
            speed[index] = float(np.linalg.norm(track[index] - track[index - 1])) / step
    speed[0] = speed[1] if len(speed) > 1 else 0.0
    return speed <= RESTING_SPEED


def _runs(flags):
    out = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _near(first, second, limit):
    return float(np.hypot(first[0] - second[0], first[1] - second[1])) <= limit
