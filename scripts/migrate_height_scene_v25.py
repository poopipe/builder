"""one-off: migrate height.scene from format 24 to 25"""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path
from typing import Any


def _handle_length() -> float:
    return 1.5


def _polyline_shape_ids(nodes: list[dict[str, Any]]) -> set[str]:
    shape_ids: set[str] = set()
    node: dict[str, Any]
    for node in nodes:
        shape: Any = node.get("vector_shape")
        if isinstance(shape, dict) and shape.get("kind") == "polyline":
            node_id: Any = node.get("id")
            if isinstance(node_id, str):
                shape_ids.add(node_id)
    return shape_ids


def _ordered_points_for_shape(
    nodes: list[dict[str, Any]], shape_id: str
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    node: dict[str, Any]
    for node in nodes:
        if node.get("parent_id") != shape_id:
            continue
        if node.get("role") not in ("vector_point", "path_point"):
            continue
        points.append(node)
    return points


def _translation(node: dict[str, Any]) -> tuple[float, float, float]:
    transform: Any = node.get("transform", {})
    translation: Any = transform.get("translation", [0.0, 0.0, 0.0])
    if not isinstance(translation, list) or len(translation) < 3:
        return (0.0, 0.0, 0.0)
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )


def _new_id(seed: str, suffix: str) -> str:
    return f"{seed}_{suffix}"


def _make_handle(
    point_id: str, role: str, name: str, offset: tuple[float, float, float]
) -> dict[str, Any]:
    return {
        "id": _new_id(point_id, role),
        "parent_id": point_id,
        "transform": {
            "translation": [offset[0], offset[1], offset[2]],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "mesh_id": None,
        "name": name,
        "role": role,
    }


def _ensure_polyline_handles(
    nodes: list[dict[str, Any]], shape_ids: set[str]
) -> list[dict[str, Any]]:
    """add default in/out handles on polyline path points that lack them"""
    extras: list[dict[str, Any]] = []
    existing_ids: set[str] = {
        str(node.get("id")) for node in nodes if isinstance(node.get("id"), str)
    }
    shape_id: str
    for shape_id in shape_ids:
        points: list[dict[str, Any]] = _ordered_points_for_shape(nodes, shape_id)
        index: int
        for index, point in enumerate(points):
            point_id: Any = point.get("id")
            if not isinstance(point_id, str):
                continue
            has_in: bool = any(
                n.get("parent_id") == point_id
                and n.get("role")
                in ("bezier_handle_in", "path_handle_in")
                for n in nodes
            )
            has_out: bool = any(
                n.get("parent_id") == point_id
                and n.get("role")
                in ("bezier_handle_out", "path_handle_out")
                for n in nodes
            )
            prev_t: tuple[float, float, float] = _translation(
                points[index - 1] if index > 0 else point
            )
            next_t: tuple[float, float, float] = _translation(
                points[index + 1] if index + 1 < len(points) else point
            )
            cur_t: tuple[float, float, float] = _translation(point)
            dx_out: float = next_t[0] - cur_t[0]
            dz_out: float = next_t[2] - cur_t[2]
            dx_in: float = prev_t[0] - cur_t[0]
            dz_in: float = prev_t[2] - cur_t[2]
            length: float = _handle_length()
            out_len: float = hypot(dx_out, dz_out)
            in_len: float = hypot(dx_in, dz_in)
            if out_len < 1e-6:
                out_offset = (length, 0.0, 0.0)
            else:
                out_offset = (
                    dx_out / out_len * length,
                    0.0,
                    dz_out / out_len * length,
                )
            if in_len < 1e-6:
                in_offset = (-length, 0.0, 0.0)
            else:
                in_offset = (
                    dx_in / in_len * length,
                    0.0,
                    dz_in / in_len * length,
                )
            if index > 0 and not has_in:
                handle = _make_handle(
                    point_id, "path_handle_in", "In", in_offset
                )
                if handle["id"] not in existing_ids:
                    extras.append(handle)
                    existing_ids.add(handle["id"])
            if index + 1 < len(points) and not has_out:
                handle = _make_handle(
                    point_id, "path_handle_out", "Out", out_offset
                )
                if handle["id"] not in existing_ids:
                    extras.append(handle)
                    existing_ids.add(handle["id"])
    return nodes + extras


def migrate_document(data: dict[str, Any]) -> dict[str, Any]:
    """return a v25 document with path_* roles and path_point recipes"""
    if int(data.get("version", 0)) != 24:
        raise ValueError(f"expected version 24, got {data.get('version')}")
    data["version"] = 25
    nodes_raw: Any = data.get("nodes", [])
    if not isinstance(nodes_raw, list):
        raise ValueError("nodes must be a list")
    nodes: list[dict[str, Any]] = [
        node for node in nodes_raw if isinstance(node, dict)
    ]
    shape_ids: set[str] = _polyline_shape_ids(nodes)
    node: dict[str, Any]
    for node in nodes:
        role: Any = node.get("role")
        if role == "vector_point":
            node["role"] = "path_point"
        elif role == "bezier_point":
            node["role"] = "path_point"
        elif role == "bezier_handle_in":
            node["role"] = "path_handle_in"
        elif role == "bezier_handle_out":
            node["role"] = "path_handle_out"
        if node.get("role") == "path_point":
            if "vector_point" in node:
                recipe: Any = node.pop("vector_point")
                if recipe is not None:
                    node["path_point"] = recipe
            if "path_point" not in node or node["path_point"] is None:
                node["path_point"] = {"width_scale": 1.0}
    data["nodes"] = _ensure_polyline_handles(nodes, shape_ids)
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
    print(f"migrated {path} to version 25")


if __name__ == "__main__":
    main()
