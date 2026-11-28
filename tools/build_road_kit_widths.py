"""scan DayZ road parts/*.p3d AABBs + snap points; write kit catalogue JSON

AABB reading is self-contained here (referenced from xml-data-viewer p3d
tools; does not import them). Stem classification uses builder.part_catalog.

Snap points are the road-tool mating corners named lb/le/pb/pe (left/right
begin/end) in the Memory LOD. ODOL stores selection→vertex indices; positions
come from that LOD's vertex table (parsed after the selection block), not a
whole-file float scan.

usage (from repo root, venv active)::

    python -m tools.build_road_kit_widths
    python -m tools.build_road_kit_widths --root P:\\DZ\\structures\\roads

writes to src/builder/heightfield/road_sections/road_kit_widths.json by default.

width is taken from straight pieces only: among size_x / size_z, the axis
closer to the stem length is longitudinal; the other is the road width.
kit width = median of those straight widths.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from builder.heightfield.road_sections.part_catalog import (
    PieceType,
    ParsedStem,
    default_parts_roots,
    parse_stem,
    parts_dir,
)

# package asset path (repo layout: tools/ → src/builder/...)
_DEFAULT_OUT: Path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "builder"
    / "heightfield"
    / "road_sections"
    / "road_kit_widths.json"
)

# --- p3d AABB (self-contained; road-scale ModelInfo / GeoLOD / MLOD Geometry) ---

_GEO_RES = 1.0e13
_GEO_RES_TOL = 1.0e10
_GEOLOD_SIZE_MAX_M = 50.0
_GEOLOD_SIZE_MIN_M = 0.05


@dataclass(frozen=True)
class P3dAabb:
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    source: str
    p3d_version: int | None
    notes: tuple[str, ...] = ()


def is_geometry_resolution(resolution: float) -> bool:
    return abs(float(resolution) - _GEO_RES) < _GEO_RES_TOL


def has_horizontal_extent(
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
) -> bool:
    """roads are often flat decals — Y may be zero; require footprint in XZ only"""
    return mx[0] > mn[0] and mx[2] > mn[2] and mx[1] >= mn[1]


def aabb_from_minmax(
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
    *,
    source: str,
    version: int | None,
    notes: tuple[str, ...] = (),
) -> P3dAabb:
    size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    return P3dAabb(
        min_xyz=mn,
        max_xyz=mx,
        size_xyz=size,
        source=source,
        p3d_version=version,
        notes=notes,
    )


def parse_odol_modelinfo(
    data: bytes,
) -> tuple[int, float, float, P3dAabb | None]:
    """return version, mem_sphere, geo_sphere, modelinfo AABB or None"""
    if len(data) < 16 or data[:4] != b"ODOL":
        raise ValueError("not an ODOL p3d")
    version = int(struct.unpack_from("<I", data, 4)[0])
    lod_count = int(struct.unpack_from("<I", data, 8)[0])
    if not (1 <= lod_count <= 64):
        raise ValueError(f"invalid LOD count {lod_count}")
    off = 12 + lod_count * 4
    if off + 12 > len(data):
        raise ValueError("truncated ODOL ModelInfo")
    _index, mem_sphere, geo_sphere = struct.unpack_from("<Iff", data, off)
    off += 12
    off += 12  # remarks, andHints, orHints
    off += 12  # geo_offset
    off += 8  # map colors
    off += 4  # ViewDensity
    if off + 24 > len(data):
        raise ValueError("truncated ODOL bbox")
    mn = struct.unpack_from("<fff", data, off)
    off += 12
    mx = struct.unpack_from("<fff", data, off)
    bbox: P3dAabb | None = None
    if has_horizontal_extent(mn, mx):
        bbox = aabb_from_minmax(mn, mx, source="odol_modelinfo", version=version)
    return version, float(mem_sphere), float(geo_sphere), bbox


def read_odol_geolod_aabb(
    data: bytes,
    *,
    size_max_m: float = _GEOLOD_SIZE_MAX_M,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """scan ODOL for Geometry-LOD Min/Max/AutoCenter matching GeoLodSphere

    Y thickness is ignored for size/aspect filters (decal roads are flat).
    """
    if len(data) < 16 or data[:4] != b"ODOL":
        raise ValueError("not an ODOL p3d")
    lod_count = int(struct.unpack_from("<I", data, 8)[0])
    if not (1 <= lod_count <= 64):
        raise ValueError(f"invalid LOD count {lod_count}")
    off = 12 + lod_count * 4
    if off + 12 > len(data):
        raise ValueError("truncated ODOL ModelInfo")
    geo = float(struct.unpack_from("<f", data, off + 8)[0])
    if geo <= 0:
        raise ValueError("GeoLodSphere missing or zero")
    target_diag = 2.0 * geo
    cands: list[
        tuple[
            float,
            float,
            float,
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    limit = len(data) - 36
    for i in range(0, limit, 4):
        vals = struct.unpack_from("<9f", data, i)
        mn = (vals[0], vals[1], vals[2])
        mx = (vals[3], vals[4], vals[5])
        ac = (vals[6], vals[7], vals[8])
        if not has_horizontal_extent(mn, mx):
            continue
        sizes = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        sx, _sy, sz = sizes
        if not (
            _GEOLOD_SIZE_MIN_M <= sx <= size_max_m
            and _GEOLOD_SIZE_MIN_M <= sz <= size_max_m
        ):
            continue
        # horizontal diagonal — Y is often ~0 for decals
        diag = math.sqrt(sx * sx + sz * sz)
        if abs(diag - target_diag) / target_diag > 0.25:
            continue
        mid = (
            (mn[0] + mx[0]) * 0.5,
            (mn[1] + mx[1]) * 0.5,
            (mn[2] + mx[2]) * 0.5,
        )
        ac_dist = math.sqrt(
            (ac[0] - mid[0]) ** 2
            + (ac[1] - mid[1]) ** 2
            + (ac[2] - mid[2]) ** 2
        )
        if ac_dist > 1.0 or not math.isfinite(ac_dist):
            continue
        aspect = max(sx, sz) / max(min(sx, sz), 1e-6)
        if aspect > 80.0:
            continue
        # prefer tighter footprints; ignore Y volume (often zero)
        area = sx * sz
        diag_err = abs(diag - target_diag)
        cands.append((diag_err, ac_dist, area, mn, mx))
    if not cands:
        raise ValueError("no GeoLodSphere-matching Geometry AABB found in ODOL")
    cands.sort(key=lambda c: (c[1] > 0.15, c[0], c[1], c[2]))
    return cands[0][3], cands[0][4]


def iter_mlod_lods(
    data: bytes,
) -> list[tuple[float, list[tuple[float, float, float]]]]:
    if len(data) < 12 or data[:4] != b"MLOD":
        raise ValueError("not an MLOD p3d")
    lod_count = int(struct.unpack_from("<I", data, 8)[0])
    pos = 12
    out: list[tuple[float, list[tuple[float, float, float]]]] = []
    for _ in range(lod_count):
        if pos + 8 > len(data):
            break
        tag = data[pos : pos + 4]
        pos += 4
        if tag != b"P3DM":
            break
        pos += 4  # version
        n_points, n_normals, n_faces = struct.unpack_from("<III", data, pos)
        pos += 12
        points: list[tuple[float, float, float]] = []
        for _p in range(n_points):
            if pos + 16 > len(data):
                break
            x, y, z = struct.unpack_from("<fff", data, pos)
            pos += 16
            points.append((float(x), float(y), float(z)))
        pos += n_normals * 12
        for _f in range(n_faces):
            if pos + 56 > len(data):
                break
            pos += 56
            n_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + n_len
            m_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + m_len
        if pos + 4 > len(data):
            break
        (resolution,) = struct.unpack_from("<f", data, pos)
        pos += 4
        out.append((float(resolution), points))
    return out


def aabb_from_points(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        raise ValueError("no points for AABB")
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for x, y, z in points:
        if abs(x) > 1e4 or abs(y) > 1e4 or abs(z) > 1e4:
            continue
        mins[0] = min(mins[0], x)
        mins[1] = min(mins[1], y)
        mins[2] = min(mins[2], z)
        maxs[0] = max(maxs[0], x)
        maxs[1] = max(maxs[1], y)
        maxs[2] = max(maxs[2], z)
    if not all(math.isfinite(v) for v in mins + maxs):
        raise ValueError("degenerate point AABB")
    mn = (mins[0], mins[1], mins[2])
    mx = (maxs[0], maxs[1], maxs[2])
    if not has_horizontal_extent(mn, mx):
        raise ValueError("empty Geometry LOD AABB")
    return mn, mx


def read_mlod_geometry_aabb(
    data: bytes,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lods = iter_mlod_lods(data)
    geo = [(res, pts) for res, pts in lods if is_geometry_resolution(res)]
    if not geo:
        raise ValueError("no Geometry LOD (res≈1e13) in MLOD")
    for _res, pts in geo:
        if pts:
            return aabb_from_points(pts)
    raise ValueError("Geometry LOD has no points")


def read_p3d_aabb(path: str | Path) -> P3dAabb:
    """preferred AABB for road sizing: ModelInfo, else GeoLOD scan, else MLOD Geometry"""
    path = Path(path)
    data = path.read_bytes()
    magic = data[:4]
    notes: list[str] = []

    if magic == b"MLOD":
        version = int(struct.unpack_from("<I", data, 4)[0])
        mn, mx = read_mlod_geometry_aabb(data)
        return aabb_from_minmax(mn, mx, source="mlod_geometry_lod", version=version)

    if magic != b"ODOL":
        raise ValueError(f"unknown p3d magic {magic!r}")

    version, _mem, geo_sphere, modelinfo = parse_odol_modelinfo(data)
    geolod: P3dAabb | None = None
    try:
        mn, mx = read_odol_geolod_aabb(data)
        geolod = aabb_from_minmax(mn, mx, source="odol_geolod_aabb", version=version)
    except ValueError as exc:
        notes.append(f"odol_geolod_aabb: {exc}")

    if modelinfo is not None:
        sx, sy, sz = modelinfo.size_xyz
        longest = max(sx, sy, sz)
        if geo_sphere > 0 and longest > 8.0 * geo_sphere:
            notes.append(
                f"odol_modelinfo looks bloated vs geo_sphere={geo_sphere:.3f}; "
                "preferring geolod if available"
            )
            if geolod is not None:
                return aabb_from_minmax(
                    geolod.min_xyz,
                    geolod.max_xyz,
                    source=geolod.source,
                    version=version,
                    notes=tuple(notes),
                )
        return aabb_from_minmax(
            modelinfo.min_xyz,
            modelinfo.max_xyz,
            source="odol_modelinfo",
            version=version,
            notes=tuple(notes),
        )

    if geolod is not None:
        return aabb_from_minmax(
            geolod.min_xyz,
            geolod.max_xyz,
            source=geolod.source,
            version=version,
            notes=tuple(notes),
        )
    raise ValueError("no plausible AABB for ODOL p3d")


# --- road-tool snap points (lb/le/pb/pe) ---

_SNAP_NAMES: tuple[str, ...] = ("lb", "le", "pb", "pe")
_SNAP_WIDTH_TOL_M: float = 0.75
_SNAP_MIN_SPAN_M: float = 1.0
# reject begin/end edges nearly parallel to travel (false vertex tables)
_SNAP_MAX_EDGE_TRAVEL_ABS_DOT: float = 0.85
_SNAP_MIN_POINT_SEP_M: float = 0.25


def _xz_dist(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return math.hypot(a[0] - b[0], a[2] - b[2])


def snap_mapping_plausible(
    mapped: dict[str, tuple[float, float, float]],
) -> bool:
    """reject false vertex-table hits that only match width by chance

    real road snaps are four distinct corners; begin/end edges sit across the
    road, not along travel. garbage recoveries often collapse lb/le to origin
    or align the begin edge with the chord.
    """
    lb = mapped["lb"]
    le = mapped["le"]
    pb = mapped["pb"]
    pe = mapped["pe"]
    pts: tuple[tuple[float, float, float], ...] = (lb, le, pb, pe)
    i: int
    j: int
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _xz_dist(pts[i], pts[j]) < _SNAP_MIN_POINT_SEP_M:
                return False
    mid_bx: float = 0.5 * (lb[0] + pb[0])
    mid_bz: float = 0.5 * (lb[2] + pb[2])
    mid_ex: float = 0.5 * (le[0] + pe[0])
    mid_ez: float = 0.5 * (le[2] + pe[2])
    travel_x: float = mid_ex - mid_bx
    travel_z: float = mid_ez - mid_bz
    travel_len: float = math.hypot(travel_x, travel_z)
    if travel_len < _SNAP_MIN_SPAN_M:
        return False
    travel_x /= travel_len
    travel_z /= travel_len

    def edge_ok(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> bool:
        ex: float = right[0] - left[0]
        ez: float = right[2] - left[2]
        elen: float = math.hypot(ex, ez)
        if elen < _SNAP_MIN_POINT_SEP_M:
            return False
        ex /= elen
        ez /= elen
        return abs(ex * travel_x + ez * travel_z) <= _SNAP_MAX_EDGE_TRAVEL_ABS_DOT

    return edge_ok(lb, pb) and edge_ok(le, pe)


_MEM_RES: float = 1.0e15
_MEM_RES_TOL: float = 1.0e14


def is_memory_resolution(resolution: float) -> bool:
    return abs(float(resolution) - _MEM_RES) < _MEM_RES_TOL


def parse_snap_selection_indices(data: bytes) -> dict[str, int] | None:
    """read Memory-LOD named selections lb/le/pb/pe → vertex indices"""
    parsed = parse_snap_selections(data)
    if parsed is None:
        return None
    return parsed[0]


def parse_snap_selections(
    data: bytes,
) -> tuple[dict[str, int], int] | None:
    """lb/le/pb/pe indices and byte offset past the selection block"""
    lb_at: int = data.find(b"lb\x00")
    if lb_at < 0:
        return None
    # expect a count ulong of 4 immediately before the first name
    if lb_at < 4 or struct.unpack_from("<I", data, lb_at - 4)[0] != 4:
        return None
    pos: int = lb_at
    out: dict[str, int] = {}
    name: str
    for name in _SNAP_NAMES:
        end: int = data.find(b"\x00", pos)
        if end < 0 or data[pos:end].decode("ascii", errors="replace") != name:
            return None
        pos = end + 1
        if pos + 23 > len(data):
            return None
        n_faces, always0 = struct.unpack_from("<II", data, pos)
        pos += 8
        is_sectional: int = int(data[pos])
        pos += 1
        n_sec, n_verts = struct.unpack_from("<II", data, pos)
        pos += 8
        if not (
            n_faces == 0
            and always0 == 0
            and is_sectional == 0
            and n_sec == 0
            and n_verts == 1
        ):
            return None
        (idx,) = struct.unpack_from("<H", data, pos)
        pos += 2
        (n_weights,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if n_weights != 0:
            return None
        out[name] = int(idx)
    return out, pos


def _point_ok(p: tuple[float, float, float]) -> bool:
    x, y, z = p
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return False
    if abs(y) > 0.5:
        return False
    return abs(x) < 200.0 and abs(z) < 200.0


def memory_lod_vertex_table_blob(data: bytes) -> bytes | None:
    """ODOL Memory LOD vertex-table bytes after the lb/le/pb/pe selections

    layout after selections: nTokens, nFrames, icon, selected, special,
    vertexBoneRefIsSimple (byte), sizeOfVertexTable (ulong, includes itself).
    """
    parsed = parse_snap_selections(data)
    if parsed is None:
        return None
    _indices, sel_end = parsed
    # 5 ulongs + 1 flag byte
    size_at: int = sel_end + 20 + 1
    if size_at + 4 > len(data):
        return None
    size_u: int = int(struct.unpack_from("<I", data, size_at)[0])
    if size_u < 8 or size_at + size_u > len(data):
        return None
    return data[size_at + 4 : size_at + size_u]


def _map_points_at(
    table: bytes,
    pos: int,
    indices: dict[str, int],
    max_i: int,
) -> dict[str, tuple[float, float, float]] | None:
    need: int = (max_i + 1) * 12
    if pos < 0 or pos + need > len(table):
        return None
    pts: list[tuple[float, float, float]] = [
        struct.unpack_from("<fff", table, pos + i * 12) for i in range(max_i + 1)
    ]
    if not all(_point_ok(p) for p in pts):
        return None
    return {n: pts[i] for n, i in indices.items()}


def recover_snap_points_from_table(
    table: bytes,
    indices: dict[str, int],
    *,
    width_m: float,
    step: int = 1,
) -> dict[str, tuple[float, float, float]] | None:
    """find indexed xyz table in bytes; step=1 required for ODOL VT alignment"""
    if width_m <= 0.5 or step < 1:
        return None
    if len(set(indices.values())) < 4:
        return None
    max_i: int = max(indices.values())
    need: int = (max_i + 1) * 12
    best: dict[str, tuple[float, float, float]] | None = None
    best_err: float = float("inf")
    pos: int
    for pos in range(0, len(table) - need + 1, step):
        mapped = _map_points_at(table, pos, indices, max_i)
        if mapped is None:
            continue
        wb: float = _xz_dist(mapped["lb"], mapped["pb"])
        we: float = _xz_dist(mapped["le"], mapped["pe"])
        if (
            abs(wb - width_m) > _SNAP_WIDTH_TOL_M
            or abs(we - width_m) > _SNAP_WIDTH_TOL_M
        ):
            continue
        if not snap_mapping_plausible(mapped):
            continue
        err: float = abs(wb - width_m) + abs(we - width_m)
        if err < best_err:
            best_err = err
            best = mapped
    return best


def recover_snap_points_from_mlod(
    data: bytes,
    indices: dict[str, int],
    *,
    width_m: float,
) -> dict[str, tuple[float, float, float]] | None:
    """index Memory LOD points directly from an MLOD p3d"""
    if data[:4] != b"MLOD":
        return None
    if width_m <= 0.5 or len(set(indices.values())) < 4:
        return None
    max_i: int = max(indices.values())
    res: float
    pts: list[tuple[float, float, float]]
    for res, pts in iter_mlod_lods(data):
        if not is_memory_resolution(res):
            continue
        if max_i >= len(pts):
            continue
        mapped: dict[str, tuple[float, float, float]] = {
            n: pts[i] for n, i in indices.items()
        }
        if not all(_point_ok(p) for p in mapped.values()):
            continue
        wb: float = _xz_dist(mapped["lb"], mapped["pb"])
        we: float = _xz_dist(mapped["le"], mapped["pe"])
        if (
            abs(wb - width_m) > _SNAP_WIDTH_TOL_M
            or abs(we - width_m) > _SNAP_WIDTH_TOL_M
        ):
            continue
        if not snap_mapping_plausible(mapped):
            continue
        return mapped
    return None


def recover_snap_points_from_indices(
    data: bytes,
    indices: dict[str, int],
    *,
    width_m: float,
) -> dict[str, tuple[float, float, float]] | None:
    """resolve Memory LOD vertex positions for snap selection indices"""
    mlod_hit = recover_snap_points_from_mlod(
        data, indices, width_m=width_m
    )
    if mlod_hit is not None:
        return mlod_hit
    blob: bytes | None = memory_lod_vertex_table_blob(data)
    if blob is not None:
        hit = recover_snap_points_from_table(
            blob, indices, width_m=width_m, step=1
        )
        if hit is not None:
            return hit
    # last resort: whole file, 1-byte step (road parts are small)
    return recover_snap_points_from_table(
        data, indices, width_m=width_m, step=1
    )


def extract_road_snap_points(
    data: bytes, *, width_m: float
) -> dict[str, tuple[float, float, float]] | None:
    """lb/le/pb/pe mating points in model space, or none if not recoverable"""
    indices: dict[str, int] | None = parse_snap_selection_indices(data)
    if indices is None:
        return None
    return recover_snap_points_from_indices(data, indices, width_m=width_m)


def read_p3d_snap_points(
    path: str | Path, *, width_m: float
) -> dict[str, tuple[float, float, float]] | None:
    """load p3d bytes and extract road snap points"""
    return extract_road_snap_points(Path(path).read_bytes(), width_m=width_m)


# --- kit width catalogue ---


@dataclass(frozen=True)
class PieceWidthRow:
    kit: str
    stem: str
    piece_type: str
    path: str
    length_m: float | None
    size_x: float
    size_y: float
    size_z: float
    width_m: float | None
    length_axis_m: float | None
    aabb_source: str
    min_x: float | None = None
    min_y: float | None = None
    min_z: float | None = None
    max_x: float | None = None
    max_y: float | None = None
    max_z: float | None = None
    snap_points: dict[str, tuple[float, float, float]] | None = None
    error: str | None = None


@dataclass(frozen=True)
class KitWidthRow:
    kit: str
    width_m: float | None
    straight_count: int
    width_samples_m: tuple[float, ...]
    notes: tuple[str, ...]


def width_from_straight_sizes(
    size_xyz: tuple[float, float, float],
    length_m: float,
) -> tuple[float, float]:
    """return (width_m, measured_length_m) from AABB sizes and declared stem length"""
    sx, _sy, sz = size_xyz
    err_x = abs(sx - length_m)
    err_z = abs(sz - length_m)
    if err_x <= err_z:
        return float(sz), float(sx)
    return float(sx), float(sz)


def scan_parts_roots(roots: Sequence[Path]) -> list[PieceWidthRow]:
    rows: list[PieceWidthRow] = []
    seen_stems: set[str] = set()
    root: Path
    for root in roots:
        parts = parts_dir(Path(root))
        if parts is None:
            continue
        path: Path
        for path in sorted(parts.glob("*.p3d")):
            stem = path.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            parsed: ParsedStem | None = parse_stem(stem)
            if parsed is None:
                continue
            try:
                aabb = read_p3d_aabb(path)
            except Exception as exc:  # noqa: BLE001 — catalogue rows capture failures
                rows.append(
                    PieceWidthRow(
                        kit=parsed.kit,
                        stem=stem,
                        piece_type=parsed.piece_type.name,
                        path=str(path),
                        length_m=(
                            float(parsed.length_m)
                            if parsed.length_m is not None
                            else None
                        ),
                        size_x=0.0,
                        size_y=0.0,
                        size_z=0.0,
                        width_m=None,
                        length_axis_m=None,
                        aabb_source="",
                        error=str(exc),
                    )
                )
                continue

            width_m: float | None = None
            length_axis_m: float | None = None
            length_declared: float | None = (
                float(parsed.length_m) if parsed.length_m is not None else None
            )
            if (
                parsed.piece_type == PieceType.straight
                and length_declared is not None
            ):
                width_m, length_axis_m = width_from_straight_sizes(
                    aabb.size_xyz, length_declared
                )

            rows.append(
                PieceWidthRow(
                    kit=parsed.kit,
                    stem=stem,
                    piece_type=parsed.piece_type.name,
                    path=str(path),
                    length_m=length_declared,
                    size_x=float(aabb.size_xyz[0]),
                    size_y=float(aabb.size_xyz[1]),
                    size_z=float(aabb.size_xyz[2]),
                    width_m=width_m,
                    length_axis_m=length_axis_m,
                    aabb_source=aabb.source,
                    min_x=float(aabb.min_xyz[0]),
                    min_y=float(aabb.min_xyz[1]),
                    min_z=float(aabb.min_xyz[2]),
                    max_x=float(aabb.max_xyz[0]),
                    max_y=float(aabb.max_xyz[1]),
                    max_z=float(aabb.max_xyz[2]),
                    snap_points=None,
                    error=None,
                )
            )

    # second pass: snap points need a width hint (piece or kit median)
    kit_widths: dict[str, float] = {
        kit.kit: kit.width_m
        for kit in aggregate_kits(rows)
        if kit.width_m is not None
    }
    filled: list[PieceWidthRow] = []
    row: PieceWidthRow
    for row in rows:
        if row.error is not None:
            filled.append(row)
            continue
        hint: float | None = row.width_m
        if hint is None:
            hint = kit_widths.get(row.kit)
        snaps: dict[str, tuple[float, float, float]] | None = None
        if hint is not None:
            try:
                snaps = read_p3d_snap_points(row.path, width_m=hint)
            except OSError:
                snaps = None
        filled.append(
            PieceWidthRow(
                kit=row.kit,
                stem=row.stem,
                piece_type=row.piece_type,
                path=row.path,
                length_m=row.length_m,
                size_x=row.size_x,
                size_y=row.size_y,
                size_z=row.size_z,
                width_m=row.width_m,
                length_axis_m=row.length_axis_m,
                aabb_source=row.aabb_source,
                min_x=row.min_x,
                min_y=row.min_y,
                min_z=row.min_z,
                max_x=row.max_x,
                max_y=row.max_y,
                max_z=row.max_z,
                snap_points=snaps,
                error=row.error,
            )
        )
    return filled


def aggregate_kits(rows: Sequence[PieceWidthRow]) -> list[KitWidthRow]:
    by_kit: dict[str, list[float]] = {}
    notes_by_kit: dict[str, list[str]] = {}
    row: PieceWidthRow
    for row in rows:
        notes_by_kit.setdefault(row.kit, [])
        if row.error:
            notes_by_kit[row.kit].append(f"{row.stem}: {row.error}")
            continue
        if row.piece_type != PieceType.straight.name:
            continue
        if row.width_m is None:
            continue
        by_kit.setdefault(row.kit, []).append(row.width_m)
        if row.length_m is not None and row.length_axis_m is not None:
            err = abs(row.length_axis_m - row.length_m)
            if err > max(0.75, 0.08 * row.length_m):
                notes_by_kit[row.kit].append(
                    f"{row.stem}: length axis {row.length_axis_m:.3f} "
                    f"vs declared {row.length_m:.0f} (Δ={err:.3f})"
                )

    kits: list[KitWidthRow] = []
    kit_name: str
    for kit_name in sorted(set(list(by_kit.keys()) + list(notes_by_kit.keys()))):
        samples = tuple(sorted(by_kit.get(kit_name, [])))
        width: float | None = None
        if samples:
            width = float(statistics.median(samples))
        kits.append(
            KitWidthRow(
                kit=kit_name,
                width_m=width,
                straight_count=len(samples),
                width_samples_m=samples,
                notes=tuple(notes_by_kit.get(kit_name, [])),
            )
        )
    return kits


def build_catalogue(roots: Sequence[Path]) -> dict[str, Any]:
    rows = scan_parts_roots(roots)
    kits = aggregate_kits(rows)
    return {
        "roots": [str(Path(r)) for r in roots],
        "kits": [asdict(k) for k in kits],
        "pieces": [asdict(r) for r in rows],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="build road kit width catalogue from p3d AABBs"
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="roads root or parts dir (repeatable); default = DayZ parts roots",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="output JSON path (default: package road_kit_widths.json)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots: tuple[Path, ...]
    if args.root:
        roots = tuple(args.root)
    else:
        roots = default_parts_roots
    catalogue = build_catalogue(roots)
    out: Path = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")

    kits = catalogue["kits"]
    pieces = catalogue["pieces"]
    with_snaps: int = sum(1 for p in pieces if p.get("snap_points"))
    print(
        f"wrote {out} ({len(kits)} kits, {len(pieces)} pieces, "
        f"{with_snaps} with snap points)"
    )
    kit: dict[str, Any]
    for kit in kits:
        width = kit["width_m"]
        width_s = f"{width:.3f} m" if width is not None else "n/a"
        print(
            f"  {kit['kit']}: width={width_s} "
            f"(straights={kit['straight_count']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
