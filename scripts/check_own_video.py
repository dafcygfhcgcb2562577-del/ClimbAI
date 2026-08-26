import sys

from climb_ai.cli import build_parser, check
from climb_ai.pose import UserError
from scripts._console import setup

СПРАВКА = """Разбор одного видео без сайта: то же самое, что команда «разбор».

    python -m scripts.check_own_video "видео.mp4"

Трассу указывать не надо: место на стене определяется раскладом зацепок.
Если базы эталонов нет, покажет только выполненные техники."""


def main():
    setup()
    if len(sys.argv) < 2:
        print(СПРАВКА)
        sys.exit(2)

    try:
        check(build_parser().parse_args(["разбор", "--видео", sys.argv[1]]))
    except UserError as ошибка:
        print(f"\n{ошибка}")
        sys.exit(2)


if __name__ == "__main__":
    main()
