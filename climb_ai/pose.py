from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_FPS = 25.0


class Lm(IntEnum):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


MIN_TORSO = 0.02

MAX_TORSO_SHARE = 0.5

OUT_OF_FRAME = 0.05

BODY = (Lm.LEFT_SHOULDER, Lm.RIGHT_SHOULDER, Lm.LEFT_HIP, Lm.RIGHT_HIP)

KEY_POINTS = (
    Lm.LEFT_SHOULDER, Lm.RIGHT_SHOULDER,
    Lm.LEFT_WRIST, Lm.RIGHT_WRIST,
    Lm.LEFT_KNEE, Lm.RIGHT_KNEE,
    Lm.LEFT_ANKLE, Lm.RIGHT_ANKLE,
)


@dataclass(frozen=True)
class Frame:
    index: int
    second: float
    points: np.ndarray
    visible: np.ndarray = None
    camera_y: float = 0.0
    aspect: float = 1.0

    def __post_init__(self):
        if self.visible is None:
            object.__setattr__(self, "visible", np.ones(len(self.points), dtype=np.float32))

    def xy(self, landmark):
        return float(self.points[landmark][0]), float(self.points[landmark][1])

    @property
    def hip_y_in_frame(self):
        return float(self.points[Lm.LEFT_HIP][1] + self.points[Lm.RIGHT_HIP][1]) / 2.0

    @property
    def hip_y(self):
        return self.hip_y_in_frame - self.camera_y


WIDE_ASPECT = 1.5

CROP_SHARE = 0.6


class PoseEngine:
    def __init__(self, model_path, max_side = 960):
        self.max_side = int(max_side)
        self._model = _read_model(str(model_path))
        self._landmarker = None
        self._searcher = None
        self.reset()

    def reset(self):
        self.close()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=self._model),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, image, at_ms):
        rgb = cv2.cvtColor(_shrink(image, self.max_side), cv2.COLOR_BGR2RGB)
        found = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(at_ms)
        )
        if found.pose_landmarks:
            return _as_arrays(found.pose_landmarks[0])
        if image.shape[1] / image.shape[0] >= WIDE_ASPECT:
            return self._detect_in_halves(image)
        return None, None

    def _detect_in_halves(self, image):
        width = image.shape[1]
        aspect = float(width) / float(image.shape[0])
        cut = int(width * CROP_SHARE)
        best = None
        for left in (0, width - cut):
            piece = _shrink(image[:, left : left + cut], self.max_side)
            rgb = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
            found = self._search().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not found.pose_landmarks:
                continue
            points, visible = _as_arrays(found.pose_landmarks[0])
            points[:, 0] = (points[:, 0] * cut + left) / width
            if best is None or _torso_of(points, aspect) > _torso_of(best[0], aspect):
                best = (points, visible)
        return best if best is not None else (None, None)

    def _search(self):
        if self._searcher is None:
            self._searcher = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_buffer=self._model),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=0.5,
                )
            )
        return self._searcher

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._searcher is not None:
            self._searcher.close()
            self._searcher = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def read_poses(
    video,
    engine,
    sample_step = 5,
    on_progress = None,
):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        if Path(video).is_file():
            raise UserError("Файл не открылся как видео: он повреждён или это не видео.")
        raise UserError(f"Видео не найдено: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    aspect = float(width) / float(height) if height else 1.0

    frames = []
    camera = CameraTrack()
    engine.reset()
    try:
        for index, image in _every_nth(capture, max(1, int(sample_step))):
            points, visible = engine.detect(image, at_ms=round(index / fps * 1000))
            if points is not None and not body_in_frame(points, aspect):
                points, visible = None, None
            camera_y = camera.update(image, points)
            if points is not None:
                frames.append(
                    Frame(
                        index=index,
                        second=index / fps,
                        points=points,
                        visible=visible,
                        camera_y=camera_y,
                        aspect=aspect,
                    )
                )
            if on_progress is not None and total > 0:
                on_progress(min(1.0, index / total))
    finally:
        capture.release()
    return frames


def _as_arrays(points):
    return (
        np.array([[p.x, p.y] for p in points], dtype=np.float32),
        np.array([p.visibility for p in points], dtype=np.float32),
    )


def _torso_of(points, aspect = 1.0):
    плечи = (points[Lm.LEFT_SHOULDER] + points[Lm.RIGHT_SHOULDER]) / 2.0
    таз = (points[Lm.LEFT_HIP] + points[Lm.RIGHT_HIP]) / 2.0
    вбок = float(плечи[0] - таз[0]) * aspect
    вверх = float(плечи[1] - таз[1])
    return float(np.hypot(вбок, вверх))


def body_in_frame(points, aspect = 1.0):
    тело = points[list(BODY)]
    if float(тело.min()) < -OUT_OF_FRAME or float(тело.max()) > 1.0 + OUT_OF_FRAME:
        return False
    return _torso_of(points, aspect) <= MAX_TORSO_SHARE


def torso_unit(frames):
    lengths = []
    for frame in frames:
        плечи = (frame.points[Lm.LEFT_SHOULDER] + frame.points[Lm.RIGHT_SHOULDER]) / 2.0
        таз = (frame.points[Lm.LEFT_HIP] + frame.points[Lm.RIGHT_HIP]) / 2.0
        lengths.append(float(np.hypot((плечи[0] - таз[0]) * frame.aspect, плечи[1] - таз[1])))
    return max(float(np.median(lengths)), 1e-3)


def poses_with_cache(video, engine, sample_step, cache):
    aspect = video_aspect(video)
    stored = Path(cache) / f"{cache_name(video)}.npz"
    if stored.is_file():
        frames = _read_cache(stored, aspect, sample_step, engine.max_side)
        if frames is not None:
            return frames
    frames = read_poses(video, engine, sample_step)
    _write_cache(stored, frames, sample_step, engine.max_side)
    return frames


def cache_name(video):
    video = Path(video)
    return f"{video.parent.name}_{video.stem}" if video.parent.name else video.stem


def video_aspect(video):
    capture = cv2.VideoCapture(str(video))
    try:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
    finally:
        capture.release()
    return float(width) / float(height) if height else 1.0


DEFAULT_ASPECT = 9 / 16


def frames_from_cache(path, aspect=None):
    saved = np.load(path)
    if aspect is None:
        aspect = float(saved["аспект"]) if "аспект" in saved.files else DEFAULT_ASPECT
    return _frames_of(saved, aspect)


def _read_cache(path, aspect, sample_step, max_side):
    saved = np.load(path)
    for name, value in (("шаг", sample_step), ("сторона", max_side)):
        if name not in saved.files:
            print(f"  кэш поз {path.name} без пометки «{name}», беру видео заново")
            return None
        if int(saved[name]) != int(value):
            print(f"  кэш поз {path.name} считан при другом значении «{name}», беру видео заново")
            return None
    return _frames_of(saved, aspect)


def _frames_of(saved, aspect):
    return [
        Frame(
            index=int(i),
            second=float(s),
            points=p,
            visible=v,
            camera_y=float(c),
            aspect=aspect,
        )
        for i, s, p, c, v in zip(
            saved["indexes"], saved["seconds"], saved["points"], saved["camera"], saved["visible"],
            strict=True,
        )
    ]


def _write_cache(path, frames, sample_step, max_side):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indexes=np.array([f.index for f in frames]),
        seconds=np.array([f.second for f in frames]),
        points=np.array([f.points for f in frames]),
        camera=np.array([f.camera_y for f in frames]),
        visible=np.array([f.visible for f in frames]),
        шаг=np.array(sample_step),
        сторона=np.array(max_side),
        аспект=np.array(frames[0].aspect),
    )


class CameraTrack:
    WIDTH = 192
    MAX_JUMP = 0.33

    def __init__(self):
        self._previous = None
        self._offset = 0.0

    def update(self, image, points = None):
        small = self._prepare(image, points)
        if self._previous is not None and self._previous.shape == small.shape:
            (_dx, dy), _confidence = cv2.phaseCorrelate(self._previous, small)
            step = float(dy) / small.shape[0]
            if abs(step) < self.MAX_JUMP:
                self._offset += step
        self._previous = small
        return self._offset

    def _prepare(self, image, points = None):
        height = max(1, round(image.shape[0] * self.WIDTH / image.shape[1]))
        grey = cv2.cvtColor(cv2.resize(image, (self.WIDTH, height)), cv2.COLOR_BGR2GRAY)
        grey = grey.astype(np.float32)
        if points is not None:
            xs, ys = points[:, 0] * self.WIDTH, points[:, 1] * height
            pad_x, pad_y = 0.25 * (xs.max() - xs.min()), 0.10 * (ys.max() - ys.min())
            left, right = int(max(0, xs.min() - pad_x)), int(min(self.WIDTH, xs.max() + pad_x))
            top, bottom = int(max(0, ys.min() - pad_y)), int(min(height, ys.max() + pad_y))
            if right > left and bottom > top:
                grey[top:bottom, left:right] = float(grey.mean())
        return grey


def read_frame_image(video, index):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = capture.read()
        return image if ok else None
    finally:
        capture.release()


def write_jpeg(path, image):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok:
        path.write_bytes(buffer.tobytes())


class UserError(Exception):
    pass


def _every_nth(capture, step):
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            return
        if index % step == 0:
            yield index, image
        index += 1


def _shrink(image, max_side):
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=1)
def _read_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise UserError(
            f"Не найдена модель позы: {path}\n"
            "Положите pose_landmarker_full.task в папку artifacts "
            "или укажите путь в переменной CLIMB_MODEL_PATH."
        )
    data = path.read_bytes()
    if not data:
        raise UserError(f"Файл модели позы пустой: {path}")
    return data


def pose_change(first, second):
    поза = _skeleton(first)
    другая = _skeleton(second)
    return float(np.linalg.norm(поза - другая, axis=1).mean())


def _skeleton(frame):
    вширь = np.array([frame.aspect, 1.0], dtype=np.float64)
    точки = frame.points * вширь
    таз = (точки[Lm.LEFT_HIP] + точки[Lm.RIGHT_HIP]) / 2.0
    плечи = (точки[Lm.LEFT_SHOULDER] + точки[Lm.RIGHT_SHOULDER]) / 2.0
    торс = max(float(np.hypot(*(плечи - таз))), MIN_TORSO)
    return (np.array([точки[point] for point in KEY_POINTS]) - таз) / торс
