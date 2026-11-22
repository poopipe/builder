"""one-off: migrate height.scene from format 22 to 23"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def migrate_document(data: dict[str, Any]) -> dict[str, Any]:
    """return a v23 document (version bump only; road source is additive)"""
    if int(data.get("version", 0)) != 22:
        raise ValueError(f"expected version 22, got {data.get('version')}")
    data["version"] = 23
    return data


def main() -> None:
    path: Path = Path(__file__).resolve().parents[1] / "height.scene"
    text: str = path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    migrated: dict[str, Any] = migrate_document(data)
    path.write_text(
        json.dumps(migrated, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"migrated {path} to version 23")


if __name__ == "__main__":
    main()
