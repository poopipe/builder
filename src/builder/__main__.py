"""package entry: ``python -m builder``"""

from __future__ import annotations

import sys

from builder.application import run_application
from builder.session_journal import log_exception


def main() -> None:
    try:
        run_application()
    except (FileNotFoundError, RuntimeError, OSError) as error:
        log_exception("application failed", error)
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
