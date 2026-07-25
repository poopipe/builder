"""package entry: ``python -m builder``"""

from __future__ import annotations

import sys

from builder.application import run_application


def main() -> None:
    try:
        run_application()
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
