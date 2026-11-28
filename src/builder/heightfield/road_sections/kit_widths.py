"""load kit width catalogue written by tools.build_road_kit_widths"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def road_kit_widths_path() -> Path:
    """package path for road_kit_widths.json"""
    return Path(__file__).with_name("road_kit_widths.json")


@dataclass(frozen=True)
class PieceSnapPoints:
    """road-tool mating corners in model space (y usually 0)"""

    lb: tuple[float, float, float]
    le: tuple[float, float, float]
    pb: tuple[float, float, float]
    pe: tuple[float, float, float]

    def begin_mid_xz(self) -> tuple[float, float]:
        """centerline at begin (lb/pb)"""
        return (
            0.5 * (self.lb[0] + self.pb[0]),
            0.5 * (self.lb[2] + self.pb[2]),
        )

    def end_mid_xz(self) -> tuple[float, float]:
        """centerline at end (le/pe)"""
        return (
            0.5 * (self.le[0] + self.pe[0]),
            0.5 * (self.le[2] + self.pe[2]),
        )

    def length_m(self) -> float:
        """begin→end centerline length in xz"""
        bx: float
        bz: float
        ex: float
        ez: float
        bx, bz = self.begin_mid_xz()
        ex, ez = self.end_mid_xz()
        dx: float = ex - bx
        dz: float = ez - bz
        return (dx * dx + dz * dz) ** 0.5


@dataclass(frozen=True)
class PieceMetrics:
    """per-stem measurements from the widths catalogue"""

    stem: str
    length_axis_m: float | None
    snap_points: PieceSnapPoints | None


def _vec3(raw: Any) -> tuple[float, float, float] | None:
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None


def _parse_snap_points(raw: Any) -> PieceSnapPoints | None:
    if not isinstance(raw, dict):
        return None
    lb = _vec3(raw.get("lb"))
    le = _vec3(raw.get("le"))
    pb = _vec3(raw.get("pb"))
    pe = _vec3(raw.get("pe"))
    if lb is None or le is None or pb is None or pe is None:
        return None
    return PieceSnapPoints(lb=lb, le=le, pb=pb, pe=pe)


def load_kit_widths(path: Path | None = None) -> dict[str, float]:
    """kit name → width_m; skips kits with no measured width"""
    json_path: Path = path if path is not None else road_kit_widths_path()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"road kit widths catalogue missing: {json_path} "
            "(run python -m tools.build_road_kit_widths)"
        )
    raw: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("road_kit_widths.json root must be an object")
    kits_raw: Any = raw.get("kits")
    if not isinstance(kits_raw, list):
        raise ValueError("road_kit_widths.json missing kits list")
    widths: dict[str, float] = {}
    entry: Any
    for entry in kits_raw:
        if not isinstance(entry, dict):
            continue
        name: Any = entry.get("kit")
        width: Any = entry.get("width_m")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(width, (int, float)):
            continue
        widths[name] = float(width)
    return widths


def load_piece_metrics(path: Path | None = None) -> dict[str, PieceMetrics]:
    """stem → measured length / snap points from the catalogue"""
    json_path: Path = path if path is not None else road_kit_widths_path()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"road kit widths catalogue missing: {json_path} "
            "(run python -m tools.build_road_kit_widths)"
        )
    raw: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("road_kit_widths.json root must be an object")
    pieces_raw: Any = raw.get("pieces")
    if not isinstance(pieces_raw, list):
        return {}
    out: dict[str, PieceMetrics] = {}
    entry: Any
    for entry in pieces_raw:
        if not isinstance(entry, dict):
            continue
        stem: Any = entry.get("stem")
        if not isinstance(stem, str) or not stem:
            continue
        length_raw: Any = entry.get("length_axis_m")
        length_axis: float | None = None
        if isinstance(length_raw, (int, float)):
            length_axis = float(length_raw)
        snaps: PieceSnapPoints | None = _parse_snap_points(
            entry.get("snap_points")
        )
        if length_axis is None and snaps is None:
            continue
        out[stem] = PieceMetrics(
            stem=stem,
            length_axis_m=length_axis,
            snap_points=snaps,
        )
    return out
