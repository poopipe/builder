"""one-off: migrate height.scene from format 21 to 22"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def migrate_vector_shape(shape: dict[str, Any], layer_flatten: float | None) -> None:
    """in-place shape migration for v22"""
    brush: str = str(shape.get("brush_kind", "add"))
    intensity_raw: Any = shape.get("intensity", 0.5)
    amount: float = float(intensity_raw)
    is_flatten: bool = brush in ("flatten", "flatten_up", "flatten_down")
    if brush == "subtract":
        shape["brush_kind"] = "add"
        amount = -amount
    if is_flatten and layer_flatten is not None:
        shape["offset"] = float(layer_flatten)
    else:
        shape["offset"] = amount
    # flatten mix; unused for add
    shape["intensity"] = float(intensity_raw) if is_flatten else 1.0


def migrate_document(data: dict[str, Any]) -> dict[str, Any]:
    """return a v22 document"""
    if int(data.get("version", 0)) != 21:
        raise ValueError(f"expected version 21, got {data.get('version')}")
    nodes: list[Any] = list(data.get("nodes", []))
    # map layer id -> flatten_offset
    layer_offsets: dict[str, float] = {}
    node: Any
    for node in nodes:
        if not isinstance(node, dict):
            continue
        layer_raw: Any = node.get("vector_layer")
        if isinstance(layer_raw, dict) and "flatten_offset" in layer_raw:
            layer_offsets[str(node["id"])] = float(layer_raw["flatten_offset"])
            node["vector_layer"] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        shape_raw: Any = node.get("vector_shape")
        if not isinstance(shape_raw, dict):
            continue
        parent_id: Any = node.get("parent_id")
        layer_flatten: float | None = None
        if isinstance(parent_id, str):
            layer_flatten = layer_offsets.get(parent_id)
        migrate_vector_shape(shape_raw, layer_flatten)
    data["version"] = 22
    data["nodes"] = nodes
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
    print(f"migrated {path} to version 22")


if __name__ == "__main__":
    main()
