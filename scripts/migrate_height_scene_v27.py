"""one-off: migrate height.scene from format 26 to 27

vector_road sources gain a coverage heightmap (absolute ribbon + coverage lerp)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
    bytes_per_sample,
)
from builder.heightfield.height_raw_cache import write_height_raw_cache


def _used_heightmap_ids(data: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    heightmaps: Any = data.get("heightmaps", [])
    if isinstance(heightmaps, list):
        entry: Any
        for entry in heightmaps:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                ids.add(str(entry["id"]))
    return ids


def _alloc_paint_id(used: set[str]) -> str:
    index: int = 1
    while True:
        candidate: str = f"paint_{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _heightmap_by_id(
    heightmaps: list[dict[str, Any]], heightmap_id: str
) -> dict[str, Any] | None:
    entry: dict[str, Any]
    for entry in heightmaps:
        if entry.get("id") == heightmap_id:
            return entry
    return None


def _write_zero_f32_hraw(path: Path, width: int, height: int) -> None:
    """create a blank f32 coverage raster beside the scene"""
    sample_type: HeightSampleType = HeightSampleType.f32
    payload: bytes = b"\x00" * (
        width * height * bytes_per_sample(sample_type)
    )
    write_height_raw_cache(
        path,
        DecodedHeightImage(
            width=width,
            height=height,
            sample_type=sample_type,
            data=payload,
        ),
    )


def migrate_document(
    data: dict[str, Any], *, scene_dir: Path | None = None
) -> dict[str, Any]:
    """return a v27 document with coverage_id on every vector_road source"""
    if int(data.get("version", 0)) != 26:
        raise ValueError(f"expected version 26, got {data.get('version')}")
    data["version"] = 27
    heightmaps_raw: Any = data.get("heightmaps", [])
    if not isinstance(heightmaps_raw, list):
        raise ValueError("heightmaps must be a list")
    heightmaps: list[dict[str, Any]] = [
        entry for entry in heightmaps_raw if isinstance(entry, dict)
    ]
    data["heightmaps"] = heightmaps
    used: set[str] = _used_heightmap_ids(data)
    nodes: Any = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    node: Any
    for node in nodes:
        if not isinstance(node, dict):
            continue
        heightfield: Any = node.get("heightfield")
        if not isinstance(heightfield, dict):
            continue
        layers: Any = heightfield.get("edit_layers", [])
        if not isinstance(layers, list):
            continue
        layer: Any
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            height: Any = layer.get("height")
            if not isinstance(height, dict):
                continue
            if height.get("source_type") != "vector_road":
                continue
            if isinstance(height.get("coverage_id"), str) and height["coverage_id"]:
                continue
            road_map_id: Any = height.get("heightmap_id")
            if not isinstance(road_map_id, str) or road_map_id == "":
                raise ValueError("vector_road.heightmap_id must be a non-empty string")
            road_entry: dict[str, Any] | None = _heightmap_by_id(
                heightmaps, road_map_id
            )
            if road_entry is None:
                raise ValueError(f"missing heightmap entry for {road_map_id}")
            width: int = int(road_entry["width"])
            height_px: int = int(road_entry["height"])
            coverage_id: str = _alloc_paint_id(used)
            height["coverage_id"] = coverage_id
            relative: str = f"heightmaps/{coverage_id}.hraw"
            heightmaps.append(
                {
                    "id": coverage_id,
                    "label": "Road Coverage",
                    "source": relative,
                    "cache": relative,
                    "sample_type": "f32",
                    "width": width,
                    "height": height_px,
                }
            )
            if scene_dir is not None:
                _write_zero_f32_hraw(
                    scene_dir / relative, width, height_px
                )
    return data


def main() -> None:
    path: Path = Path(__file__).resolve().parents[1] / "height.scene"
    text: str = path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    migrated: dict[str, Any] = migrate_document(data, scene_dir=path.parent)
    path.write_text(
        json.dumps(migrated, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"migrated {path} to version 27")


if __name__ == "__main__":
    main()
