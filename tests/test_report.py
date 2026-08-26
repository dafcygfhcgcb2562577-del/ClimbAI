from climb_ai.analyze import Report, Spot
from climb_ai.report import as_text, describe, summary
from climb_ai.technique import Moment, Technique


def момент():
    return Moment(Technique.TWIST, start_sec=1.0, end_sec=3.0, from_progress=0.1, to_progress=0.2)


def место(done, at_sec = 25.0):
    return Spot(
        at_sec=at_sec,
        frame_index=int(at_sec * 25),
        techniques=[Technique.FLAG_LEFT],
        done=done,
        image="miss_01.jpg",
    )


def test_пролаз_не_найден_говорит_про_съёмку():
    отчёт = Report(video="v", climb_sec=(0.0, 0.0), note="Подъёма по стене не видно.")

    assert summary(отчёт)["verdict"] == "нет данных"


def test_без_эталона_показываем_выполненное_а_не_ошибку_разбора():
    отчёт = Report(video="v", climb_sec=(1.0, 20.0), done=[момент()], note="Базы эталонов нет.")

    итог = summary(отчёт)
    assert итог["verdict"] == "только выполненное"
    assert "не разобран" not in итог["headline"].lower()


def test_проверенный_пролаз_без_ошибок_это_не_то_же_что_без_эталона():
    без_эталона = Report(video="v", climb_sec=(1.0, 20.0), done=[момент()])
    нечего_требовать = Report(video="v", climb_sec=(1.0, 20.0), done=[момент()], checked=True)
    чисто = Report(
        video="v", climb_sec=(1.0, 20.0), done=[момент()], checked=True, spots=[место(done=True)]
    )

    assert summary(без_эталона)["verdict"] == "только выполненное"
    assert summary(нечего_требовать)["verdict"] == "не с чем сравнить"
    assert summary(чисто)["verdict"] == "хорошо"


def test_места_делятся_на_сделанное_и_нет():
    отчёт = Report(
        video="v",
        climb_sec=(1.0, 20.0),
        checked=True,
        spots=[место(done=True, at_sec=5.0), место(done=False, at_sec=9.0), место(done=False, at_sec=14.0)],
    )

    assert len(отчёт.correct) == 1
    assert len(отчёт.mistakes) == 2
    assert summary(отчёт)["headline"] == "2 ошибки"


def test_ошибки_считаются_и_склоняются():
    отчёт = Report(video="v", climb_sec=(1.0, 20.0), checked=True, spots=[место(False)])
    assert summary(отчёт)["headline"] == "1 ошибка"

    отчёт.spots = [место(False, at_sec=float(i)) for i in range(3)]
    assert summary(отчёт)["headline"] == "3 ошибки"

    отчёт.spots = [место(False, at_sec=float(i)) for i in range(5)]
    assert summary(отчёт)["headline"] == "5 ошибок"


def test_место_описывается_временем_и_техникой_без_лишнего():
    текст = describe(место(done=False))

    assert текст == "0:25 Флажок (левая нога)"


def test_в_отчёте_нет_длинных_тире_и_пояснений():
    отчёт = Report(
        video="v",
        climb_sec=(1.0, 20.0),
        checked=True,
        spots=[место(done=True, at_sec=5.0), место(done=False, at_sec=9.0)],
    )

    текст = as_text(отчёт)

    assert "—" not in текст
    assert "–" not in текст
    assert len(текст.splitlines()) <= 8, "отчёт должен быть коротким"


def test_кадры_есть_и_у_сделанного_и_у_ошибок():
    отчёт = Report(
        video="v",
        climb_sec=(1.0, 20.0),
        checked=True,
        spots=[место(done=True, at_sec=5.0), место(done=False, at_sec=9.0)],
    )

    assert all(s.image for s in отчёт.spots)


def test_нет_требований_это_не_похвала():
    нечего = Report(video="v", climb_sec=(1.0, 20.0), checked=True, spots=[],
                    note="На этих зацепках эталоны не сходятся.")
    чисто = Report(video="v", climb_sec=(1.0, 20.0), checked=True, spots=[место(done=True)])

    assert summary(нечего)["verdict"] == "не с чем сравнить"
    assert summary(чисто)["verdict"] == "хорошо"


def test_каждая_техника_описана_на_вкладке():
    from climb_ai.lessons import as_list
    from climb_ai.technique import TECHNIQUES

    уроки = as_list()

    assert {у["название"] for у in уроки} == set(TECHNIQUES)
    assert all(у["суть"] and у["как"] for у in уроки), "у каждой техники есть суть и шаги"


def test_ролики_техник_на_месте():
    from climb_ai.lessons import LESSONS, VIDEO_DIR_NAME
    from climb_ai.settings import PROJECT_ROOT

    папка = PROJECT_ROOT / VIDEO_DIR_NAME
    if not папка.is_dir():
        return
    for урок in LESSONS:
        if урок.video:
            assert (папка / урок.video).is_file(), f"нет ролика {урок.video}"


def test_страницы_сайта_открываются():
    from fastapi.testclient import TestClient

    from web.backend.main import app

    клиент = TestClient(app)

    assert клиент.get("/").status_code == 200
    assert клиент.get("/техники").status_code == 200
    assert клиент.get("/job/нет-такого").status_code == 404


def test_на_страницах_нет_длинных_тире_и_версии_кода():
    from fastapi.testclient import TestClient

    from web.backend.main import app

    клиент = TestClient(app)

    for адрес in ("/", "/техники"):
        страница = клиент.get(адрес).text
        assert "—" not in страница, f"длинное тире на {адрес}"
        assert "Работает код" not in страница, f"версия кода на {адрес}"
