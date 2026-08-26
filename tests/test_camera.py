import cv2
import numpy as np

from climb_ai.pose import CameraTrack, Frame, Lm
from climb_ai.route import find_climb


def стена(высота_сдвига, размер = 240):
    большая = np.zeros((размер * 3, размер, 3), dtype=np.uint8)
    камни = np.random.default_rng(7).integers(0, 255, size=(60, 3))
    места = np.random.default_rng(11).integers(10, размер * 3 - 10, size=(60, 2))
    for (y, x), цвет in zip(места, камни, strict=True):
        cv2.circle(большая, (int(x), int(y)), 7, tuple(int(c) for c in цвет), -1)
    верх = размер + высота_сдвига
    return большая[верх : верх + размер]


def test_неподвижная_камера_даёт_нулевой_сдвиг():
    камера = CameraTrack()
    кадр = стена(0)

    сдвиги = [камера.update(кадр) for _ in range(5)]

    assert all(abs(с) < 0.01 for с in сдвиги)


def test_камера_едет_вверх_и_это_видно():
    камера = CameraTrack()
    сдвиги = [камера.update(стена(-шаг * 8)) for шаг in range(6)]

    assert сдвиги[-1] > 0.1, "подъём камеры должен накапливаться"
    assert all(b >= a - 1e-6 for a, b in zip(сдвиги[:-1], сдвиги[1:], strict=True))


def test_резкий_скачок_не_накапливается():
    камера = CameraTrack()
    камера.update(стена(0))
    камера.update(стена(-4))
    до_скачка = камера._offset

    камера.update(np.random.default_rng(3).integers(0, 255, size=(240, 240, 3), dtype=np.uint8))

    assert abs(камера._offset - до_скачка) < CameraTrack.MAX_JUMP


def кадр_на_высоте(index, second, таз_в_кадре, камера):
    points = np.zeros((33, 2), dtype=np.float32)
    points[Lm.LEFT_SHOULDER] = (0.46, таз_в_кадре - 0.20)
    points[Lm.RIGHT_SHOULDER] = (0.54, таз_в_кадре - 0.20)
    points[Lm.LEFT_HIP] = (0.47, таз_в_кадре)
    points[Lm.RIGHT_HIP] = (0.53, таз_в_кадре)
    points[Lm.LEFT_KNEE] = (0.47, таз_в_кадре + 0.18)
    points[Lm.RIGHT_KNEE] = (0.53, таз_в_кадре + 0.18)
    for точка in (Lm.LEFT_ANKLE, Lm.LEFT_HEEL, Lm.LEFT_FOOT_INDEX):
        points[точка] = (0.47, таз_в_кадре + 0.36)
    for точка in (Lm.RIGHT_ANKLE, Lm.RIGHT_HEEL, Lm.RIGHT_FOOT_INDEX):
        points[точка] = (0.53, таз_в_кадре + 0.36)
    return Frame(index=index, second=second, points=points, camera_y=камера)


def test_пролаз_находится_когда_камеру_ведут_за_спортсменом():
    кадры = [
        кадр_на_высоте(i * 5, i * 0.2, таз_в_кадре=0.55, камера=i * 0.02)
        for i in range(40)
    ]

    пролаз = find_climb(кадры)

    assert пролаз is not None
    assert len(пролаз) > 30, "в пролаз должно попасть почти всё видео"
    assert пролаз.progress[0] < 0.2
    assert пролаз.progress[-1] > 0.8


def test_без_движения_камеры_поведение_прежнее():
    кадры = [
        кадр_на_высоте(i * 5, i * 0.2, таз_в_кадре=0.85 - i * 0.016, камера=0.0)
        for i in range(40)
    ]

    пролаз = find_climb(кадры)

    assert пролаз is not None
    assert пролаз.progress[0] < 0.2
    assert пролаз.progress[-1] > 0.8
