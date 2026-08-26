import argparse
import io
import json
import math
import sys
from pathlib import Path

from climb_ai.analyze import analyze
from climb_ai.pose import PoseEngine, UserError
from climb_ai.reference import (
    Reference,
    build,
    build_from_cache,
    drop_useless,
    gain_of,
    has_reference,
    reference_path,
    sources_of,
)
from climb_ai.report import as_text, save_images
from climb_ai.settings import settings

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV")

MIN_COMPARISONS = 30


def noise_of(agreement, chance, tries):
    if tries <= 0:
        return 1.0
    первая = agreement * (1.0 - agreement) / tries
    вторая = chance * (1.0 - chance) / tries
    return math.sqrt(первая + вторая)


def looks_like_chance(reference):
    tries = reference.tries()
    if tries < MIN_COMPARISONS:
        return True
    выигрыш = reference.agreement() - reference.by_chance()
    шум = noise_of(reference.agreement(), reference.by_chance(), tries)
    return выигрыш <= 2.0 * шум


def _videos_in(folder):
    if not folder.is_dir():
        raise UserError(f"Папка с видео не найдена: {folder}")
    found = [path for path in folder.iterdir() if path.suffix in VIDEO_SUFFIXES]
    if not found:
        raise UserError(f"В папке {folder} нет видеофайлов")
    return sorted(found)


def select_useful(reference):
    было = gain_of(reference)
    убранные = []
    отобранный = drop_useless(reference, lambda имя, старое, новое: убранные.append(имя))
    if not убранные:
        return reference
    print()
    print(f"Отбор: выигрыш над случайным {было:+.1%} стал {gain_of(отобранный):+.1%}, "
          f"пролазов осталось {len(sources_of(отобранный))}.")
    for имя in убранные:
        print(f"  убран: {имя}")
    return отобранный


def make_reference(args):
    cache = Path(args.кэш) if args.кэш else None
    print("Нужны ОБРАЗЦОВЫЕ пролазы: база отвечает на вопрос «как надо», а не «как обычно».\n")

    if not args.видео:
        if cache is None:
            raise UserError("Укажите --видео с папкой пролазов или --кэш с разобранными позами.")
        print(f"Собираю базу из разобранных поз: {cache}\n")
        reference = build_from_cache(cache)
    else:
        videos = _videos_in(Path(args.видео))
        print(f"Собираю базу эталонов из {len(videos)} видео.\n")
        with PoseEngine(settings.pose_model, settings.pose_max_side) as engine:
            reference = build(videos, engine, settings.sample_step, cache)

    reference = select_useful(reference)

    path = reference_path()
    reference.save(path)
    agreement = reference.agreement()
    print(f"\nБаза сохранена: {path}")
    с_техникой = sum(1 for example in reference.examples if example.active)
    print(f"Пролазов: {reference.total}. Моментов: {len(reference.examples)}, "
          f"из них с техникой: {с_техникой}.")
    выигрыш = agreement - reference.by_chance()
    шум = noise_of(agreement, reference.by_chance(), reference.tries())
    print(f"Согласие эталонов между собой: {agreement:.0%}, "
          f"случайное совпадение: {reference.by_chance():.0%}.")
    print(f"Выигрыш над случайным: {выигрыш:+.1%} при пределе шума "
          f"{2.0 * шум:.1%} и {reference.tries()} сравнениях.")
    if looks_like_chance(reference):
        print()
        print("Выигрыш не отличим от шума: пролазы в базе проходят одинаковые")
        print("зацепки по-разному. Требование сложится только там, где половина")
        print("соседей сойдётся на одной технике, и таких мест будет мало.")
        print("Добавьте пролазы одного уровня и одного стиля.")


def check(args):
    video = Path(args.видео)
    if not video.is_file():
        raise UserError(f"Видео не найдено: {video}")

    reference = Reference.load(reference_path()) if has_reference() else None
    if reference is None:
        print("Базы эталонов нет — покажу только выполненные техники, без разбора ошибок.")
        print('Собрать: python app.py эталон --видео "папка с образцовыми пролазами"\n')

    output = Path(args.куда) if args.куда else settings.artifacts / "разбор" / video.stem
    output.mkdir(parents=True, exist_ok=True)

    with PoseEngine(settings.pose_model, settings.pose_max_side) as engine:
        report = analyze(video, engine, reference, settings.sample_step, _print_progress)

    save_images(report, video, output)
    (output / "отчёт.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + as_text(report))
    print(f"\nОтчёт и кадры: {output}")


def show_reference(_args):
    if not has_reference():
        print("Базы эталонов пока нет. Соберите:")
        print('  python app.py эталон --видео "папка с образцовыми пролазами"')
        return
    reference = Reference.load(reference_path())
    print(f"База эталонов: {reference_path()}")
    print(f"  пролазов: {reference.total}")
    for name in reference.climbs:
        print(f"    {name}")
    активных = sum(1 for example in reference.examples if example.active)
    print(f"  моментов: {len(reference.examples)}, из них с техникой: {активных}")
    counts = {}
    for example in reference.examples:
        for name in example.techniques:
            counts[name] = counts.get(name, 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: {count}")
    выигрыш = reference.agreement() - reference.by_chance()
    шум = noise_of(reference.agreement(), reference.by_chance(), reference.tries())
    print(f"  согласие эталонов между собой: {reference.agreement():.0%}, "
          f"случайное совпадение: {reference.by_chance():.0%}")
    print(f"  выигрыш над случайным: {выигрыш:+.1%} при пределе шума "
          f"{2.0 * шум:.1%} и {reference.tries()} сравнениях")
    if looks_like_chance(reference):
        print("  ВНИМАНИЕ: выигрыш не отличим от шума, база почти ничего не потребует")


_last_shown = [-1]


def _print_progress(stage, part):
    percent = int(part * 100)
    if percent == _last_shown[0]:
        return
    _last_shown[0] = percent
    print(f"\r  {stage}... {percent}%   ", end="", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="ClimbAI: разбор техники лазания по видео.",
    )
    commands = parser.add_subparsers(dest="команда", required=True)

    reference = commands.add_parser("эталон", help="собрать базу из образцовых пролазов")
    reference.add_argument("--видео", default="", help="папка с видео образцовых пролазов")
    reference.add_argument(
        "--кэш", default="",
        help="папка с разобранными позами: пересборка за секунды, видео не нужны",
    )
    reference.set_defaults(run=make_reference)

    check_cmd = commands.add_parser("разбор", help="разобрать пролаз и найти ошибки")
    check_cmd.add_argument("--видео", required=True, help="видеофайл пролаза")
    check_cmd.add_argument("--куда", default="", help="папка для отчёта и кадров")
    check_cmd.set_defaults(run=check)

    show = commands.add_parser("эталон-состав", help="показать, что лежит в базе эталонов")
    show.set_defaults(run=show_reference)
    return parser


def _speak_russian_in_console():
    for имя in ("stdout", "stderr"):
        поток = getattr(sys, имя)
        if getattr(поток, "encoding", "").lower().replace("-", "") == "utf8":
            continue
        буфер = getattr(поток, "buffer", None)
        if буфер is not None:
            setattr(
                sys, имя,
                io.TextIOWrapper(буфер, encoding="utf-8", errors="replace", line_buffering=True),
            )


def run():
    _speak_russian_in_console()
    args = build_parser().parse_args()
    try:
        args.run(args)
    except UserError as error:
        print(f"\n{error}")
        sys.exit(2)
