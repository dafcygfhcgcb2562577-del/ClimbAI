from dataclasses import dataclass

import numpy as np

from climb_ai.pose import Frame, Lm, torso_unit

MIN_CLIMB_RISE = 0.7

MIN_FEET_SHARE = 0.4

SMOOTH_FRAMES = 5


@dataclass(frozen=True)
class Climb:
    frames: list[Frame]
    progress: np.ndarray

    @property
    def seconds(self):
        return self.frames[0].second, self.frames[-1].second

    def __len__(self):
        return len(self.frames)


NEAR_TOP = 0.2

NEAR_BOTTOM = 0.05


def find_climb(frames):
    if len(frames) < 10:
        return None

    unit = torso_unit(frames)
    height = _smooth(np.array([f.hip_y for f in frames], dtype=np.float64))
    start_at, end_at = _biggest_rise(height)
    end_at = _while_at_top(height, end_at, float(height[start_at] - height[end_at]))

    inside = height[start_at : end_at + 1]
    bottom, top = float(np.percentile(inside, 95)), float(np.percentile(inside, 2))
    rise = bottom - top
    if rise / unit < MIN_CLIMB_RISE:
        return None

    start_at = _while_at_bottom(height, start_at, end_at, bottom - NEAR_BOTTOM * rise)
    if end_at - start_at < 5:
        return None
    if _feet_rise(frames[start_at : end_at + 1]) < MIN_FEET_SHARE * rise:
        return None

    chosen = frames[start_at : end_at + 1]
    climbed = (bottom - height[start_at : end_at + 1]) / rise
    return Climb(frames=chosen, progress=np.clip(climbed, 0.0, 1.0))


def _feet_rise(frames):
    feet = np.array(
        [
            max(float(f.points[Lm.LEFT_ANKLE][1]), float(f.points[Lm.RIGHT_ANKLE][1])) - f.camera_y
            for f in frames
        ]
    )
    return float(np.percentile(feet, 95) - np.percentile(feet, 2))


def _biggest_rise(height):
    низ = best = 0
    top = 0
    for index in range(1, len(height)):
        if height[index] > height[низ]:
            низ = index
        elif height[низ] - height[index] > height[best] - height[top]:
            best, top = низ, index
    return best, top


def _while_at_bottom(height, start_at, end_at, limit):
    while start_at + 1 < end_at and height[start_at + 1] >= limit:
        start_at += 1
    return start_at


def _while_at_top(height, end_at, rise):
    limit = height[end_at] + NEAR_TOP * rise
    while end_at + 1 < len(height) and height[end_at + 1] <= limit:
        end_at += 1
    return end_at


def _smooth(values, window = SMOOTH_FRAMES):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]
