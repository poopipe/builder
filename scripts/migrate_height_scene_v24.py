"""one-off: migrate height.scene from format 23 to 24"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def migrate_document(data: dict[str, Any]) -> dict[str, Any]:
    """return a v24 document with vector_point recipes on point nodes"""
    if int(data.get("version", 0)) != 23:
        raise ValueError(f"expected version 23, got {data.get('version')}")
    data["version"] = 24
    nodes: Any = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    node: Any
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("role") != "vector_point":
            continue
        if "vector_point" not in node or node["vector_point"] is None:
            node["vector_point"] = {"width_scale": 1.0}
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
    print(f"migrated {path} to version 24")


if __name__ == "__main__":
    main()
