import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from climb_ai.holds import LEGS
from climb_ai.pose import MIN_TORSO, Lm


class Technique(StrEnum):
    TWIST = "Скручивание"
    FLAG_LEFT = "Флажок (левая нога)"
    FLAG_RIGHT = "Флажок (правая нога)"
    SPREAD = "Распор"


TECHNIQUES = [t.value for t in Technique]

MIN_SCORE = 0.25


@dataclass(frozen=True)
class Pose:
    torso: float
    hip_turn: float
    shoulder_shift: float
    knee_angle: tuple[float, float]
    foot_side: tuple[float, float]
    foot_lift: tuple[float, float]
    feet_apart: float
    feet_height_gap: float
    leg_seen: tuple[float, float] = (1.0, 1.0)
    body_seen: float = 1.0

    @classmethod
    def of(
        cls, points, visible = None, aspect = 1.0
    ):
        if aspect != 1.0:
            points = points * np.array([aspect, 1.0], dtype=points.dtype)
        left_hip, right_hip = points[Lm.LEFT_HIP], points[Lm.RIGHT_HIP]
        left_shoulder, right_shoulder = points[Lm.LEFT_SHOULDER], points[Lm.RIGHT_SHOULDER]
        left_knee, right_knee = points[Lm.LEFT_KNEE], points[Lm.RIGHT_KNEE]
        left_foot, right_foot = _foot(points, "left"), _foot(points, "right")

        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        torso = max(float(np.hypot(*(shoulder_center - hip_center))), MIN_TORSO)
        shoulder_width = max(abs(float(left_shoulder[0] - right_shoulder[0])), 1e-6)
        hip_width = abs(float(left_hip[0] - right_hip[0]))

        if visible is None:
            visible = np.ones(len(points), dtype=np.float32)
        leg_seen = (
            _grows(min(visible[[Lm.LEFT_KNEE, Lm.LEFT_ANKLE, Lm.LEFT_FOOT_INDEX]]), 0.4, 0.3),
            _grows(min(visible[[Lm.RIGHT_KNEE, Lm.RIGHT_ANKLE, Lm.RIGHT_FOOT_INDEX]]), 0.4, 0.3),
        )
        body_seen = _grows(
            min(visible[[Lm.LEFT_SHOULDER, Lm.RIGHT_SHOULDER, Lm.LEFT_HIP, Lm.RIGHT_HIP]]),
            0.4, 0.3,
        )

        return cls(
            torso=torso,
            leg_seen=leg_seen,
            body_seen=body_seen,
            hip_turn=hip_width / shoulder_width,
            shoulder_shift=abs(float(shoulder_center[0] - hip_center[0])) / torso,
            knee_angle=(
                _angle(left_hip, left_knee, points[Lm.LEFT_ANKLE]),
                _angle(right_hip, right_knee, points[Lm.RIGHT_ANKLE]),
            ),
            foot_side=(
                abs(float(left_foot[0] - hip_center[0])) / torso,
                abs(float(right_foot[0] - hip_center[0])) / torso,
            ),
            foot_lift=(
                float(right_foot[1] - left_foot[1]) / torso,
                float(left_foot[1] - right_foot[1]) / torso,
            ),
            feet_apart=abs(float(left_foot[0] - right_foot[0])) / torso,
            feet_height_gap=abs(float(left_foot[1] - right_foot[1])) / torso,
        )


def score_all(
    points, visible = None, aspect = 1.0
):
    pose = Pose.of(points, visible, aspect)
    return {
        Technique.TWIST: _twist(pose),
        Technique.FLAG_LEFT: _flag(pose, 0),
        Technique.FLAG_RIGHT: _flag(pose, 1),
        Technique.SPREAD: _spread(pose),
    }


def detect(points, min_score = MIN_SCORE):
    scores = score_all(points)
    name, score = max(scores.items(), key=lambda item: item[1])
    return (name, score) if score >= min_score else (None, score)


SMOOTH_SEC = 0.5


@dataclass
class Moment:
    technique: str
    start_sec: float
    end_sec: float
    from_progress: float
    to_progress: float
    peak_sec: float = 0.0
    peak_index: int = 0

    @property
    def seconds(self):
        return self.end_sec - self.start_sec


def performed(frames, progress, contacts=None, smooth_sec = SMOOTH_SEC):
    if not len(frames):
        return []

    smoothed = smoothed_scores(frames, smooth_sec, contacts)
    seconds = [float(frame.second) for frame in frames]
    places = [float(place) for place in progress]

    found = []
    for column, name in enumerate(TECHNIQUES):
        for start, stop in _runs(smoothed[:, column] >= MIN_SCORE):
            peak = start + int(np.argmax(smoothed[start : stop + 1, column]))
            found.append(
                Moment(
                    technique=name,
                    start_sec=seconds[start],
                    end_sec=seconds[stop],
                    from_progress=places[start],
                    to_progress=places[stop],
                    peak_sec=seconds[peak],
                    peak_index=int(frames[peak].index),
                )
            )
    return sorted(found, key=lambda moment: moment.start_sec)


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


def smoothed_scores(frames, smooth_sec = SMOOTH_SEC, contacts=None):
    if not len(frames):
        return np.zeros((0, len(TECHNIQUES)))
    seconds = np.array([frame.second for frame in frames], dtype=np.float64)
    scores = np.array(
        [
            [score_all(frame.points, frame.visible, frame.aspect)[name] for name in TECHNIQUES]
            for frame in frames
        ]
    )
    if contacts is not None:
        scores = scores * _foot_is_free(frames, contacts)
    return _average_over_time(scores, seconds, smooth_sec)


FLAG_COLUMNS = (TECHNIQUES.index(Technique.FLAG_LEFT), TECHNIQUES.index(Technique.FLAG_RIGHT))

def _foot_is_free(frames, contacts):
    free = np.ones((len(frames), len(TECHNIQUES)))
    for row, frame in enumerate(frames):
        for column, leg in zip(FLAG_COLUMNS, LEGS, strict=True):
            standing = any(
                contact.limb == leg and contact.holds_at(frame.second) for contact in contacts
            )
            free[row, column] = 0.0 if standing else 1.0
    return free


def can_be_seen(points, visible, aspect):
    pose = Pose.of(points, visible, aspect)
    legs = min(pose.leg_seen)
    return {
        Technique.TWIST: pose.body_seen,
        Technique.FLAG_LEFT: legs,
        Technique.FLAG_RIGHT: legs,
        Technique.SPREAD: legs,
    }


def techniques_of(row):
    return frozenset(
        name for index, name in enumerate(TECHNIQUES) if row[index] >= MIN_SCORE
    )


def _average_over_time(scores, seconds, window_sec):
    if window_sec <= 0 or len(scores) < 2:
        return scores
    near = np.abs(seconds[:, None] - seconds[None, :]) <= window_sec / 2.0
    return (near @ scores) / near.sum(axis=1, keepdims=True)


def _twist(pose):
    turned = _falls(pose.hip_turn, 0.58, 0.28)
    reaching = _grows(pose.shoulder_shift, 0.08, 0.17)
    return _all_of(turned, reaching, pose.body_seen)


def _flag(pose, side):
    straight = _grows(pose.knee_angle[side], 156.0, 17.0)
    aside = _grows(pose.foot_side[side], 0.25, 0.25)
    weightless = _grows(pose.foot_lift[side], 0.0, 0.45)
    return _all_of(straight, aside, weightless, pose.leg_seen[side], pose.leg_seen[1 - side])


def _spread(pose):
    wide = _grows(pose.feet_apart, 0.90, 0.51)
    level = _falls(pose.feet_height_gap, 0.8, 0.5)
    loaded = _falls(max(pose.knee_angle), 170.0, 27.0)
    return _all_of(wide, level, loaded, *pose.leg_seen)


def _all_of(*conditions):
    lowest = min(conditions)
    if lowest <= 0.0:
        return 0.0
    return float(np.exp(np.mean(np.log(conditions))))


def _grows(value, start, span):
    return float(np.clip((value - start) / span, 0.0, 1.0))


def _falls(value, start, span):
    return float(np.clip((start - value) / span, 0.0, 1.0))


def _foot(points, side):
    parts = (
        (Lm.LEFT_ANKLE, Lm.LEFT_HEEL, Lm.LEFT_FOOT_INDEX)
        if side == "left"
        else (Lm.RIGHT_ANKLE, Lm.RIGHT_HEEL, Lm.RIGHT_FOOT_INDEX)
    )
    return max((points[part] for part in parts), key=lambda point: float(point[1]))


def _angle(first, corner, second):
    a, b = first - corner, second - corner
    cosine = float(a @ b) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
