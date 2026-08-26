import numpy as np
import pytest

from climb_ai.pose import Lm
from climb_ai.technique import TECHNIQUES, Pose, Technique, detect, performed, score_all


def стоящий_ровно():
    points = np.zeros((33, 2), dtype=np.float32)
    points[Lm.LEFT_SHOULDER] = (0.46, 0.30)
    points[Lm.RIGHT_SHOULDER] = (0.54, 0.30)
    points[Lm.LEFT_HIP] = (0.47, 0.50)
    points[Lm.RIGHT_HIP] = (0.53, 0.50)
    points[Lm.LEFT_KNEE] = (0.47, 0.68)
    points[Lm.RIGHT_KNEE] = (0.53, 0.68)
    points[Lm.LEFT_ANKLE] = (0.47, 0.86)
    points[Lm.RIGHT_ANKLE] = (0.53, 0.86)
    points[Lm.LEFT_HEEL] = (0.47, 0.87)
    points[Lm.RIGHT_HEEL] = (0.53, 0.87)
    points[Lm.LEFT_FOOT_INDEX] = (0.47, 0.88)
    points[Lm.RIGHT_FOOT_INDEX] = (0.53, 0.88)
    points[Lm.LEFT_WRIST] = (0.45, 0.14)
    points[Lm.RIGHT_WRIST] = (0.55, 0.14)
    return points


def сдвинуть_и_уменьшить(points, dx, dy, scale):
    middle = points.mean(axis=0)
    return (points - middle) * scale + middle + np.array([dx, dy], dtype=np.float32)


def test_поза_не_зависит_от_места_и_крупности_в_кадре():
    рядом = стоящий_ровно()
    далеко = сдвинуть_и_уменьшить(рядом, dx=-0.25, dy=0.15, scale=0.35)

    у_первого = score_all(рядом)
    у_второго = score_all(далеко)
    for техника in TECHNIQUES:
        assert у_первого[техника] == pytest.approx(у_второго[техника], abs=0.02), техника


def test_ровная_поза_не_считается_техникой():
    название, _оценка = detect(стоящий_ровно())
    assert название is None


def test_распор_узнаётся_по_широко_разведённым_ногам():
    поза = стоящий_ровно()
    поза[Lm.LEFT_ANKLE] = поза[Lm.LEFT_HEEL] = поза[Lm.LEFT_FOOT_INDEX] = (0.30, 0.78)
    поза[Lm.RIGHT_ANKLE] = поза[Lm.RIGHT_HEEL] = поза[Lm.RIGHT_FOOT_INDEX] = (0.70, 0.78)
    поза[Lm.LEFT_KNEE] = (0.32, 0.62)
    поза[Lm.RIGHT_KNEE] = (0.68, 0.62)

    assert detect(поза)[0] == Technique.SPREAD


def test_прямые_разведённые_ноги_это_не_распор():
    поза = стоящий_ровно()
    поза[Lm.LEFT_ANKLE] = поза[Lm.LEFT_HEEL] = поза[Lm.LEFT_FOOT_INDEX] = (0.30, 0.78)
    поза[Lm.RIGHT_ANKLE] = поза[Lm.RIGHT_HEEL] = поза[Lm.RIGHT_FOOT_INDEX] = (0.70, 0.78)
    поза[Lm.LEFT_KNEE] = (0.385, 0.64)
    поза[Lm.RIGHT_KNEE] = (0.615, 0.64)

    assert score_all(поза)[Technique.SPREAD] == 0.0


def test_флажок_узнаётся_по_прямой_отведённой_ноге():
    поза = стоящий_ровно()
    поза[Lm.RIGHT_KNEE] = (0.55, 0.66)
    поза[Lm.RIGHT_ANKLE] = поза[Lm.RIGHT_HEEL] = поза[Lm.RIGHT_FOOT_INDEX] = (0.52, 0.80)
    поза[Lm.LEFT_KNEE] = (0.36, 0.62)
    поза[Lm.LEFT_ANKLE] = поза[Lm.LEFT_HEEL] = поза[Lm.LEFT_FOOT_INDEX] = (0.24, 0.74)

    assert detect(поза)[0] == Technique.FLAG_LEFT


def test_скручивание_узнаётся_по_развороту_таза():
    поза = стоящий_ровно()
    поза[Lm.LEFT_HIP] = (0.500, 0.50)
    поза[Lm.RIGHT_HIP] = (0.522, 0.50)
    поза[Lm.LEFT_SHOULDER] = (0.40, 0.30)
    поза[Lm.RIGHT_SHOULDER] = (0.48, 0.30)

    assert detect(поза)[0] == Technique.TWIST


def test_оценки_лежат_от_нуля_до_единицы():
    for поза in (стоящий_ровно(), сдвинуть_и_уменьшить(стоящий_ровно(), 0.1, -0.1, 2.0)):
        for название, оценка in score_all(поза).items():
            assert 0.0 <= оценка <= 1.0, название


class ПоддельныйКадр:
    def __init__(self, second, points, visible = None):
        self.index = int(second * 25)
        self.second = second
        self.points = points
        self.visible = np.ones(len(points), dtype=np.float32) if visible is None else visible
        self.aspect = 1.0


def test_одиночный_всплеск_не_становится_техникой():
    распор = стоящий_ровно()
    распор[Lm.LEFT_ANKLE] = распор[Lm.LEFT_HEEL] = распор[Lm.LEFT_FOOT_INDEX] = (0.30, 0.78)
    распор[Lm.RIGHT_ANKLE] = распор[Lm.RIGHT_HEEL] = распор[Lm.RIGHT_FOOT_INDEX] = (0.70, 0.78)
    распор[Lm.LEFT_KNEE] = (0.36, 0.66)
    распор[Lm.RIGHT_KNEE] = (0.64, 0.66)

    кадры = [ПоддельныйКадр(t, стоящий_ровно()) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
    кадры[2] = ПоддельныйКадр(0.4, распор)
    найдено = performed(кадры, np.linspace(0, 1, len(кадры)))

    assert найдено == []


def test_торс_нулевой_длины_не_роняет_расчёт():
    поза = стоящий_ровно()
    поза[Lm.LEFT_SHOULDER] = поза[Lm.LEFT_HIP]
    поза[Lm.RIGHT_SHOULDER] = поза[Lm.RIGHT_HIP]

    геометрия = Pose.of(поза)
    assert геометрия.torso > 0
    assert all(np.isfinite(оценка) for оценка in score_all(поза).values())


def распор_поза():
    поза = стоящий_ровно()
    поза[Lm.LEFT_ANKLE] = поза[Lm.LEFT_HEEL] = поза[Lm.LEFT_FOOT_INDEX] = (0.30, 0.78)
    поза[Lm.RIGHT_ANKLE] = поза[Lm.RIGHT_HEEL] = поза[Lm.RIGHT_FOOT_INDEX] = (0.70, 0.78)
    поза[Lm.LEFT_KNEE] = (0.32, 0.62)
    поза[Lm.RIGHT_KNEE] = (0.68, 0.62)
    return поза


def test_невидимая_нога_не_даёт_техники():
    поза = распор_поза()
    видно = np.ones(33, dtype=np.float32)
    assert score_all(поза, видно)[Technique.SPREAD] > 0.2, "при видимых ногах распор есть"

    видно[[Lm.RIGHT_ANKLE, Lm.RIGHT_FOOT_INDEX, Lm.RIGHT_KNEE]] = 0.1
    оценки = score_all(поза, видно)
    assert оценки[Technique.SPREAD] == 0.0, "ноги не видно — распора нет"
    assert оценки[Technique.FLAG_RIGHT] == 0.0
    assert оценки[Technique.FLAG_LEFT] == 0.0, "флажок судят по обеим ногам"


def test_видимость_не_мешает_когда_всё_видно():
    поза = распор_поза()
    полная = score_all(поза, np.ones(33, dtype=np.float32))
    без_видимости = score_all(поза)

    assert полная == без_видимости
