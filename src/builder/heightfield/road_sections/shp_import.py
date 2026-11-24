"""read osm/gtt road shapefiles into usage-grouped polylines"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, cast

import shapefile

from builder.heightfield.road_sections.road_sections_types import (
    classify_road_usage,
)

# douglas-peucker simplification before building path points (metres)
simplify_eps_m: float = 4.0

# shapefile types: polyline / polylinez / polylinem
_polyline_types: frozenset[int] = frozenset({3, 13, 23})


@dataclass(frozen=True)
class ImportedRoadPart:
    """one polyline part after classify + simplify"""

    usage: str
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class ShpImportResult:
    """features read from one shapefile"""

    parts: list[ImportedRoadPart]
    features_read: int
    features_used: int
    features_skipped: int


def dedupe_pts(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """drop consecutive duplicate vertices"""
    out: list[tuple[float, float]] = []
    point: tuple[float, float]
    for point in pts:
        if not out or (
            abs(point[0] - out[-1][0]) > 1e-6 or abs(point[1] - out[-1][1]) > 1e-6
        ):
            out.append(point)
    return out


def perp_dist(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """distance from point to segment ab"""
    ax: float
    ay: float
    ax, ay = a
    bx: float
    by: float
    bx, by = b
    px: float
    py: float
    px, py = point
    dx: float = bx - ax
    dy: float = by - ay
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return hypot(px - ax, py - ay)
    t: float = max(
        0.0,
        min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)),
    )
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify_polyline(
    pts: list[tuple[float, float]], eps: float
) -> list[tuple[float, float]]:
    """douglas-peucker; keeps endpoints"""
    if len(pts) < 3:
        return pts

    def rec(lo: int, hi: int) -> list[int]:
        if hi <= lo + 1:
            return []
        max_d: float = -1.0
        max_i: int = lo + 1
        i: int
        for i in range(lo + 1, hi):
            d: float = perp_dist(pts[i], pts[lo], pts[hi])
            if d > max_d:
                max_d = d
                max_i = i
        if max_d <= eps:
            return []
        return rec(lo, max_i) + [max_i] + rec(max_i, hi)

    keep: list[int] = [0] + rec(0, len(pts) - 1) + [len(pts) - 1]
    return [pts[i] for i in keep]


def field_index(fields: list[Any], *names: str) -> int | None:
    """0-based record index for the first matching field name"""
    wanted: set[str] = {name.lower() for name in names}
    index: int
    # fields[0] is deletion flag; real fields start at 1
    for index in range(1, len(fields)):
        name: str = str(fields[index][0]).strip().lower()
        if name in wanted:
            return index - 1
    return None


def record_string(record: Sequence[Any], index: int | None) -> str:
    """string attribute at record index, or empty"""
    if index is None or index < 0 or index >= len(record):
        return ""
    value: Any = record[index]
    if value is None:
        return ""
    return str(value)


def xy_from_point(point: Any) -> tuple[float, float]:
    """take the first two coordinates from a pyshp point tuple"""
    return (float(point[0]), float(point[1]))


def read_road_shp(shp_path: Path) -> ShpImportResult:
    """load polyline features, classify usage, simplify parts"""
    reader = shapefile.Reader(str(shp_path))
    try:
        fields: list[Any] = list(reader.fields)
        layer_index: int | None = field_index(fields, "GTT_road", "__LAYER")
        highway_index: int | None = field_index(fields, "HIGHWAY", "highway")
        parts: list[ImportedRoadPart] = []
        features_read: int = 0
        features_used: int = 0
        features_skipped: int = 0
        for shape_record in reader.iterShapeRecords():
            features_read += 1
            shape = shape_record.shape
            if shape is None:
                features_skipped += 1
                continue
            shape_type: int = int(shape.shapeType)
            if shape_type not in _polyline_types:
                features_skipped += 1
                continue
            record = shape_record.record
            if record is None:
                features_skipped += 1
                continue
            usage: str | None = classify_road_usage(
                record_string(cast(Sequence[Any], record), layer_index),
                record_string(cast(Sequence[Any], record), highway_index),
            )
            if usage is None:
                features_skipped += 1
                continue
            used_part: bool = False
            part_starts: list[int] = [int(start) for start in shape.parts]
            point_count: int = len(shape.points)
            part_index: int
            for part_index, start in enumerate(part_starts):
                end: int
                if part_index + 1 < len(part_starts):
                    end = part_starts[part_index + 1]
                else:
                    end = point_count
                raw_pts: list[tuple[float, float]] = [
                    xy_from_point(point)
                    for point in shape.points[start:end]
                ]
                cleaned: list[tuple[float, float]] = simplify_polyline(
                    dedupe_pts(raw_pts), simplify_eps_m
                )
                if len(cleaned) < 2:
                    continue
                parts.append(ImportedRoadPart(usage=usage, points=cleaned))
                used_part = True
            if used_part:
                features_used += 1
            else:
                features_skipped += 1
        return ShpImportResult(
            parts=parts,
            features_read=features_read,
            features_used=features_used,
            features_skipped=features_skipped,
        )
    finally:
        reader.close()
