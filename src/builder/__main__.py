"""Package entry: ``python -m builder``."""

from builder.app import App


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
