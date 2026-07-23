"""Package entry: ``python -m builder``."""

from builder.app import Application


def main() -> None:
    Application().run()


if __name__ == "__main__":
    main()
