"""one-off: migrate height.scene from format 25 to 26"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def migrate_document(data: dict[str, Any]) -> dict[str, Any]:
    """return a v26 document with grade on vector shapes"""
    if int(data.get("version", 0)) != 25:
        raise ValueError(f"expected version 25, got {data.get('version')}")
    data["version"] = 26
    nodes: Any = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    node: Any
    for node in nodes:
        if not isinstance(node, dict):
            continue
        shape: Any = node.get("vector_shape")
        if not isinstance(shape, dict):
            continue
        if "grade" not in shape:
            shape["grade"] = 1.0
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
    print(f"migrated {path} to version 26")


if __name__ == "__main__":
    main()
