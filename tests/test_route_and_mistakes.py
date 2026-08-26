import numpy as np

from climb_ai.analyze import _find_spots
from climb_ai.holds import Situation, find_contacts, hold_map, situations
from climb_ai.pose import Frame, Lm
from climb_ai.reference import Example, Reference
from climb_ai.route import find_climb
from climb_ai.technique import Moment, Technique


def кадр(
    index,
    second,
    таз,
    руки = 0.35,
    ноги = 0.0,
    ноги_внизу = None,
):
    points = np.zeros((33, 2), dtype=np.float32)
    points[Lm.LEFT_SHOULDER] = (0.46, таз - 0.20)
    points[Lm.RIGHT_SHOULDER] = (0.54, таз - 0.20)
    points[Lm.LEFT_HIP] = (0.47, таз)
    points[Lm.RIGHT_HIP] = (0.53, таз)
    points[Lm.LEFT_WRIST] = (0.5 - руки / 2, таз - 0.30)
    points[Lm.RIGHT_WRIST] = (0.5 + руки / 2, таз - 0.30)
    points[Lm.LEFT_KNEE] = (0.47, таз + 0.18)
    points[Lm.RIGHT_KNEE] = (0.53, таз + 0.18)
    низ = таз + 0.36 if ноги_внизу is None else ноги_внизу
    for точка in (Lm.LEFT_ANKLE, Lm.LEFT_HEEL, Lm.LEFT_FOOT_INDEX):
        points[точка] = (0.47 - ноги, низ)
    for точка in (Lm.RIGHT_ANKLE, Lm.RIGHT_HEEL, Lm.RIGHT_FOOT_INDEX):
        points[точка] = (0.53 + ноги, низ)
    return Frame(index=index, second=second, points=points, aspect=0.5625)


def подъём(шагов = 30, сверху = 0.2, снизу = 0.85):
    высоты = np.linspace(снизу, сверху, шагов)
    return [кадр(i * 5, i * 0.2, float(h)) for i, h in enumerate(высоты)]


def test_пролаз_находится_и_продвижение_идёт_от_нуля_до_единицы():
    пролаз = find_climb(подъём())

    assert пролаз is not None
    assert пролаз.progress[0] < 0.2
    assert пролаз.progress[-1] > 0.8


def test_подход_и_спуск_в_пролаз_не_попадают():
    стоял = [кадр(i * 5, i * 0.2, 0.85) for i in range(10)]
    лез = подъём(20)
    спускался = [кадр(500 + i * 5, 20 + i * 0.2, 0.3 + i * 0.05) for i in range(10)]

    пролаз = find_climb(стоял + [кадр(f.index + 100, f.second + 2, f.hip_y) for f in лез] + спускался)

    assert пролаз is not None
    assert len(пролаз) < len(стоял) + len(лез) + len(спускался)


def test_стоящий_на_месте_человек_пролазом_не_считается():
    assert find_climb([кадр(i * 5, i * 0.2, 0.8) for i in range(30)]) is None


def test_короткое_видео_не_ломает_расчёт():
    assert find_climb([кадр(0, 0.0, 0.8), кадр(5, 0.2, 0.7)]) is None


def test_замершая_конечность_считается_хватом():
    кадры = [кадр(i * 5, i * 0.2, 0.8 - i * 0.001) for i in range(20)]

    хваты = find_contacts(кадры)

    assert хваты, "неподвижные руки и ноги должны дать хваты"
    assert {х.limb for х in хваты} == {"левая рука", "правая рука", "левая нога", "правая нога"}


def test_быстрый_перехват_хватом_не_считается():
    кадры = []
    for i in range(12):
        f = кадр(i * 5, i * 0.2, 0.8)
        f.points[Lm.RIGHT_WRIST] = (0.2 + 0.5 * (i % 2), 0.5)
        кадры.append(f)

    правая = [х for х in find_contacts(кадры) if х.limb == "правая рука"]

    assert not правая


def test_одна_зацепка_взятая_дважды_остаётся_одной():
    кадры = [кадр(i * 5, i * 0.2, 0.8) for i in range(30)]

    карта = hold_map(find_contacts(кадры))

    assert len(карта) <= 4, f"четыре конечности на месте — не больше четырёх зацепок, а не {len(карта)}"


def test_расклад_считается_когда_обе_руки_держатся():
    кадры = [кадр(i * 5, i * 0.2, 0.8 - i * 0.001) for i in range(20)]

    расклады = situations(кадры, find_contacts(кадры))

    assert расклады
    assert all(р.hand_span[0] > 0 for р in расклады), "правая рука правее левой"


def test_расклад_не_зависит_от_крупности_съёмки():
    мелко = [кадр(i * 5, i * 0.2, 0.8 - i * 0.001) for i in range(20)]
    крупно = []
    for f in мелко:
        точки = f.points.copy()
        точки = (точки - 0.5) * 2.0 + 0.5
        крупно.append(Frame(index=f.index, second=f.second, points=точки, aspect=f.aspect))

    первый = situations(мелко, find_contacts(мелко))[0]
    второй = situations(крупно, find_contacts(крупно))[0]

    assert первый.distance_to(второй) < 0.2


def расклад(second, руки=(0.6, 0.0), лево=(-0.5, 1.2), право=(0.5, 1.2)):
    return Situation(
        second=second, frame_index=int(second * 25), hand_span=руки,
        left_foot=лево, right_foot=право,
    )


def эталон_из(*техники, сколько = 5):
    ref = Reference(climbs=[f"пролаз{i}" for i in range(сколько)])
    ref.examples = [
        Example(
            situation=расклад(second=float(i)),
            techniques=frozenset(техники),
            source=f"пролаз{i}",
        )
        for i in range(сколько)
    ]
    return ref


def test_ошибка_находится_когда_нужная_техника_не_выполнена():
    места = _find_spots([расклад(10.0)], done=[], reference=эталон_из(Technique.TWIST))

    assert len(места) == 1
    assert места[0].techniques == [Technique.TWIST]
    assert места[0].done is False


def test_на_месте_показывается_одна_техника():
    ref = эталон_из(Technique.TWIST, Technique.FLAG_LEFT)

    места = _find_spots([расклад(10.0)], done=[], reference=ref)

    assert len(места) == 1
    assert len(места[0].techniques) == 1, "две техники сразу человеку не показываем"


def test_показывается_то_что_не_сделано_а_не_то_что_сделано():
    сделал = [Moment(Technique.TWIST, start_sec=9.5, end_sec=10.5, from_progress=0.3, to_progress=0.4)]

    места = _find_spots(
        [расклад(10.0)], done=сделал, reference=эталон_из(Technique.TWIST, Technique.FLAG_LEFT)
    )

    assert len(места) == 1
    assert места[0].techniques == [Technique.FLAG_LEFT]
    assert места[0].done is False


def test_выполненная_техника_попадает_в_сделанное_а_не_в_ошибки():
    сделал = [Moment(Technique.TWIST, start_sec=9.5, end_sec=10.5, from_progress=0.3, to_progress=0.4)]

    места = _find_spots([расклад(10.0)], done=сделал, reference=эталон_из(Technique.TWIST))

    assert len(места) == 1
    assert места[0].done is True


def test_несогласный_эталон_ничего_не_требует():
    ref = Reference(climbs=["a", "b", "c", "d", "e"])
    ref.examples = [
        Example(situation=расклад(float(i)),
                techniques=frozenset({[Technique.TWIST, Technique.SPREAD, Technique.FLAG_LEFT][i % 3]}),
                source=f"пролаз{i}")
        for i in range(6)
    ]

    assert _find_spots([расклад(10.0)], done=[], reference=ref) == []


def test_непохожий_расклад_не_притягивает_эталон():
    далёкий = расклад(10.0, руки=(3.0, 2.0), лево=(-3.0, 3.0), право=(3.0, 3.0))

    assert _find_spots([далёкий], done=[], reference=эталон_из(Technique.TWIST)) == []


def test_подряд_идущие_одинаковые_требования_схлопываются():
    подряд = [расклад(10.0 + i * 0.2) for i in range(15)]

    места = _find_spots(подряд, done=[], reference=эталон_из(Technique.TWIST))

    assert len(места) <= 2, f"одно место, а не {len(места)} подряд"


def test_согласие_эталонов_считается_и_видно():
    согласный = эталон_из(Technique.TWIST)
    assert согласный.agreement() > 0.9

    разный = Reference(climbs=["a", "b", "c"])
    разный.examples = [
        Example(situation=расклад(float(i)),
                techniques=frozenset({[Technique.TWIST, Technique.SPREAD, Technique.FLAG_LEFT][i % 3]}),
                source=f"пролаз{i}")
        for i in range(6)
    ]
    assert разный.agreement() < 0.5


def test_эталон_сохраняется_и_читается_обратно(tmp_path):
    было = эталон_из(Technique.SPREAD)
    путь = tmp_path / "эталон.json"

    было.save(путь)
    стало = Reference.load(путь)

    assert стало.total == было.total
    assert len(стало.examples) == len(было.examples)
    assert стало.examples[0].techniques == frozenset({Technique.SPREAD})
    assert стало.examples[0].situation.hand_span == было.examples[0].situation.hand_span


def test_один_эталон_с_техникой_требования_не_создаёт():
    ref = Reference(climbs=["a", "b", "c"])
    ref.examples = [
        Example(situation=расклад(0.0), techniques=frozenset({Technique.TWIST}), source="a")
    ] + [
        Example(situation=расклад(float(i)), techniques=frozenset(), source="b")
        for i in range(1, 8)
    ]

    assert _find_spots([расклад(10.0)], done=[], reference=ref) == []


def test_редкая_техника_среди_многих_бездействий_не_требуется():
    ref = Reference(climbs=[f"пролаз{i}" for i in range(5)])
    ref.examples = [
        Example(situation=расклад(0.0), techniques=frozenset({Technique.TWIST}), source="a"),
        Example(situation=расклад(0.1), techniques=frozenset({Technique.TWIST}), source="b"),
        Example(situation=расклад(0.2), techniques=frozenset({Technique.TWIST}), source="c"),
    ] + [
        Example(situation=расклад(float(i)), techniques=frozenset(), source=f"пролаз{i}")
        for i in range(3, 30)
    ]

    assert _find_spots([расклад(10.0)], done=[], reference=ref) == []


def test_техника_нужна_когда_её_делают_разные_пролазы():
    ref = Reference(climbs=["a", "b", "c", "d"])
    ref.examples = [
        Example(situation=расклад(float(i)), techniques=frozenset({Technique.TWIST}),
                source=f"пролаз{i}")
        for i in range(3)
    ] + [
        Example(situation=расклад(float(i)), techniques=frozenset(), source=f"пролаз{i}")
        for i in range(3, 7)
    ]

    места = _find_spots([расклад(10.0)], done=[], reference=ref)

    assert len(места) == 1
    assert места[0].techniques == [Technique.TWIST]


def test_другая_высота_трассы_это_другое_место():
    внизу = расклад(10.0)
    наверху = Situation(
        second=40.0, frame_index=1000, hand_span=(0.6, 0.0),
        left_foot=(-0.5, 1.2), right_foot=(0.5, 1.2), progress=0.95,
    )

    assert _find_spots([внизу], done=[], reference=эталон_из(Technique.TWIST))
    assert _find_spots([наверху], done=[], reference=эталон_из(Technique.TWIST)) == []


def test_обе_ноги_флажком_одновременно_невозможны():
    from climb_ai.technique import Technique as T
    from climb_ai.technique import score_all

    поза = np.zeros((33, 2), dtype=np.float32)
    поза[Lm.LEFT_SHOULDER] = (0.40, 0.30)
    поза[Lm.RIGHT_SHOULDER] = (0.60, 0.30)
    поза[Lm.LEFT_HIP] = (0.45, 0.50)
    поза[Lm.RIGHT_HIP] = (0.55, 0.50)
    поза[Lm.LEFT_KNEE] = (0.25, 0.70)
    поза[Lm.RIGHT_KNEE] = (0.75, 0.70)
    for точка in (Lm.LEFT_ANKLE, Lm.LEFT_HEEL, Lm.LEFT_FOOT_INDEX):
        поза[точка] = (0.10, 0.90)
    for точка in (Lm.RIGHT_ANKLE, Lm.RIGHT_HEEL, Lm.RIGHT_FOOT_INDEX):
        поза[точка] = (0.90, 0.90)

    оценки = score_all(поза)
    сработали = [оценки[T.FLAG_LEFT] > 0.0, оценки[T.FLAG_RIGHT] > 0.0]

    assert not all(сработали), "обе ноги не могут висеть противовесом разом"


def test_работа_у_верхней_точки_не_отрезается():
    лез = [кадр(i * 5, i * 0.2, float(h)) for i, h in enumerate(np.linspace(0.85, 0.45, 20))]
    висел_наверху = [кадр((20 + i) * 5, (20 + i) * 0.2, 0.45 - 0.03 * (i == 12)) for i in range(25)]
    слез = [кадр((45 + i) * 5, (45 + i) * 0.2, 0.5 + i * 0.05) for i in range(10)]

    пролаз = find_climb(лез + висел_наверху + слез)

    assert пролаз is not None
    assert len(пролаз) >= 40, f"работа у верха должна остаться в пролазе, а не {len(пролаз)} кадров"
    assert пролаз.frames[-1].second <= слез[1].second, "спуск в пролаз попадать не должен"


def test_стоящий_на_земле_не_считается_пролазом():
    присел = [кадр(i * 5, i * 0.2, float(таз), ноги_внизу=0.92)
              for i, таз in enumerate(np.linspace(0.72, 0.50, 25))]

    assert find_climb(присел) is None


def test_чистый_пролаз_не_получает_придирок():
    from climb_ai.analyze import Spot, _hide_nitpicks

    места = [Spot(at_sec=float(i), frame_index=i, techniques=[Technique.TWIST], done=i < 8)
             for i in range(10)]

    оставили = _hide_nitpicks(места)

    assert all(м.done for м in оставили), "у чистого пролаза ошибок быть не должно"
    assert len(оставили) == 8


def test_пролаз_с_ошибками_придирки_сохраняет():
    from climb_ai.analyze import Spot, _hide_nitpicks

    места = [Spot(at_sec=float(i), frame_index=i, techniques=[Technique.TWIST], done=i < 3)
             for i in range(10)]

    assert len(_hide_nitpicks(места)) == 10


def test_выполненное_рядом_снимает_требование():
    from climb_ai.analyze import Spot, _drop_satisfied

    места = [
        Spot(at_sec=27.0, frame_index=675, techniques=[Technique.SPREAD, Technique.FLAG_RIGHT], done=True),
        Spot(at_sec=27.4, frame_index=685, techniques=[Technique.SPREAD, Technique.FLAG_LEFT], done=False),
    ]

    оставили = _drop_satisfied(места)

    ошибки = [м for м in оставили if not м.done]
    assert len(ошибки) == 1
    assert ошибки[0].techniques == [Technique.FLAG_LEFT], "распор уже засчитан рядом"


def test_полностью_закрытое_требование_исчезает():
    from climb_ai.analyze import Spot, _drop_satisfied

    места = [
        Spot(at_sec=10.0, frame_index=250, techniques=[Technique.TWIST], done=True),
        Spot(at_sec=10.5, frame_index=262, techniques=[Technique.TWIST], done=False),
    ]

    assert all(м.done for м in _drop_satisfied(места))


def test_два_флажка_сразу_не_требуются():
    from climb_ai.analyze import _one_flag

    было = [(Technique.FLAG_LEFT, 0.8), (Technique.FLAG_RIGHT, 0.5), (Technique.SPREAD, 0.6)]

    стало = _one_flag(было)

    assert (Technique.FLAG_RIGHT, 0.5) not in стало
    assert (Technique.FLAG_LEFT, 0.8) in стало
    assert (Technique.SPREAD, 0.6) in стало, "распор с флажком совмещается, его не трогаем"


def test_соседние_места_схлопываются_в_одно():
    from climb_ai.analyze import Spot, _thin_out

    места = [
        Spot(at_sec=20.0, frame_index=500, techniques=[Technique.FLAG_LEFT], done=False),
        Spot(at_sec=20.6, frame_index=515, techniques=[Technique.FLAG_RIGHT], done=False),
    ]

    оставили = _thin_out(места, {})

    assert len(оставили) == 1
    assert оставили[0].techniques == [Technique.FLAG_LEFT]


def test_кадры_отбираются_по_смене_позы_а_не_по_времени():
    from climb_ai.analyze import Spot, _thin_out

    def кадр(index, second, подъём, нога_вбок=0.0):
        points = np.zeros((33, 2), dtype=np.float32)
        points[Lm.LEFT_SHOULDER] = (0.45, 0.40)
        points[Lm.RIGHT_SHOULDER] = (0.55, 0.40)
        points[Lm.LEFT_HIP] = (0.46, 0.60)
        points[Lm.RIGHT_HIP] = (0.54, 0.60)
        points[Lm.LEFT_WRIST] = (0.45, 0.25)
        points[Lm.RIGHT_WRIST] = (0.55, 0.25)
        points[Lm.LEFT_KNEE] = (0.44 - нога_вбок, 0.75)
        points[Lm.RIGHT_KNEE] = (0.56, 0.75)
        points[Lm.LEFT_ANKLE] = (0.43 - нога_вбок, 0.90)
        points[Lm.RIGHT_ANKLE] = (0.57, 0.90)
        return Frame(index=index, second=second, points=points, camera_y=подъём, aspect=0.5625)

    стоит = {
        500: кадр(500, 20.0, 0.0),
        800: кадр(800, 32.0, 0.0),
    }
    места = [
        Spot(at_sec=20.0, frame_index=500, techniques=[Technique.FLAG_LEFT], done=False),
        Spot(at_sec=32.0, frame_index=800, techniques=[Technique.FLAG_LEFT], done=False),
    ]
    assert len(_thin_out(места, стоит)) == 1, "неподвижная поза даёт один кадр"

    двигался = {
        500: кадр(500, 20.0, 0.0),
        800: кадр(800, 32.0, -0.30, нога_вбок=0.80),
    }
    assert len(_thin_out(места, двигался)) == 2, "поза сменилась, нужны оба кадра"

    уехала_камера = {
        500: кадр(500, 20.0, 0.0),
        800: кадр(800, 32.0, -0.30),
    }
    assert len(_thin_out(места, уехала_камера)) == 1, "движение камеры позой не считается"


def test_показываем_не_больше_пяти_мест_каждого_вида():
    from climb_ai.analyze import MAX_SHOWN, Spot, _best_of

    места = [
        Spot(at_sec=float(i), frame_index=i, techniques=[Technique.TWIST],
             done=i % 2 == 0, share=0.5 + i / 100)
        for i in range(20)
    ]

    оставили = _best_of(места)

    assert len([м for м in оставили if not м.done]) == MAX_SHOWN
    assert len([м for м in оставили if м.done]) == MAX_SHOWN
    assert оставили == sorted(оставили, key=lambda s: s.at_sec), "по времени, а не по уверенности"


def test_оставляем_самые_уверенные():
    from climb_ai.analyze import Spot, _best_of

    места = [
        Spot(at_sec=1.0, frame_index=25, techniques=[Technique.TWIST], done=False, share=0.9),
        *[Spot(at_sec=float(i + 2), frame_index=i, techniques=[Technique.SPREAD],
               done=False, share=0.51) for i in range(8)],
    ]

    оставили = _best_of(места)

    assert any(м.share == 0.9 for м in оставили), "самое уверенное место должно остаться"
