"""diagnose consecutive road-piece junctions after fit / place"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, radians
from pathlib import Path

from builder.heightfield.road_sections.fit import curve_merge_span_m
from builder.heightfield.road_sections.part_catalog import (
    ParsedStem,
    PieceType,
    RoadKit,
    parse_stem,
    resolve_kit,
)
from builder.heightfield.road_sections.place import placements_for_polyline
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadUsageGroupRecipe,
)
from builder.session_journal import default_logs_dir

# junction clearance outside this band is flagged
overlap_tol_m: float = 0.5
gap_tol_m: float = 0.75
fit_report_filename: str = "fit_report.txt"


def default_fit_report_path() -> Path:
    """fit report path under the process logs directory"""
    return default_logs_dir() / fit_report_filename


@dataclass(frozen=True)
class FitReportRow:
    """one placement slot and the clearance ahead to the next piece"""

    polyline_id: str
    index: int
    stem: str
    role: str
    x: float
    z: float
    yaw_deg: float
    gap_ahead_m: float | None
    flags: tuple[str, ...]


def piece_role(parsed: ParsedStem | None) -> str:
    """short role label for a parsed stem"""
    if parsed is None:
        return "unknown"
    if parsed.piece_type == PieceType.straight:
        return "straight"
    if parsed.piece_type == PieceType.corner:
        return "corner"
    if parsed.piece_type == PieceType.terminator:
        return "terminator"
    if parsed.piece_type == PieceType.curve:
        return "curve"
    return "unknown"


def half_extent_m(parsed: ParsedStem | None) -> float:
    """approx half length along the centerline from pivot to a junction"""
    if parsed is None:
        return 0.0
    if parsed.piece_type == PieceType.straight and parsed.length_m is not None:
        return float(parsed.length_m) * 0.5
    if (
        parsed.piece_type in (PieceType.corner, PieceType.curve)
        and parsed.angle_deg is not None
        and parsed.radius_m is not None
    ):
        return parsed.radius_m * radians(parsed.angle_deg) * 0.5
    if parsed.piece_type == PieceType.terminator:
        return 3.0
    return 0.0


def flags_for_gap(
    gap_m: float,
    *,
    merge_span: float,
    role: str,
    next_role: str,
) -> tuple[str, ...]:
    """classify junction clearance between two consecutive pieces"""
    out: list[str] = []
    if gap_m < -overlap_tol_m:
        out.append("overlap")
    elif gap_m > gap_tol_m:
        out.append("gap")
        if gap_m < merge_span:
            out.append("short")
    if role == "corner" and next_role == "corner" and 0.05 < gap_m < merge_span:
        out.append("unmerged")
    return tuple(out)


def report_polyline_placements(
    polyline_id: str,
    placements: Sequence[RoadPiecePlacement],
    kit: RoadKit,
) -> list[FitReportRow]:
    """build junction rows for one polyline's placements in order"""
    merge_span: float = curve_merge_span_m(kit)
    rows: list[FitReportRow] = []
    index: int
    for index, placement in enumerate(placements):
        parsed: ParsedStem | None = parse_stem(placement.stem)
        role: str = piece_role(parsed)
        gap_ahead: float | None = None
        flags: tuple[str, ...] = ()
        if index + 1 < len(placements):
            nxt: RoadPiecePlacement = placements[index + 1]
            next_parsed: ParsedStem | None = parse_stem(nxt.stem)
            next_role: str = piece_role(next_parsed)
            pivot_span: float = hypot(nxt.x - placement.x, nxt.z - placement.z)
            gap_ahead = (
                pivot_span - half_extent_m(parsed) - half_extent_m(next_parsed)
            )
            flags = flags_for_gap(
                gap_ahead,
                merge_span=merge_span,
                role=role,
                next_role=next_role,
            )
        rows.append(
            FitReportRow(
                polyline_id=polyline_id,
                index=index,
                stem=placement.stem,
                role=role,
                x=placement.x,
                z=placement.z,
                yaw_deg=placement.yaw_deg,
                gap_ahead_m=gap_ahead,
                flags=flags,
            )
        )
    return rows


def report_usage_group_polylines(
    group: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
    kit: RoadKit,
) -> list[FitReportRow]:
    """report placements for the given polyline ids on a usage group"""
    rows: list[FitReportRow] = []
    polyline_id: str
    for polyline_id in polyline_ids:
        rows.extend(
            report_polyline_placements(
                polyline_id,
                placements_for_polyline(group, polyline_id),
                kit,
            )
        )
    return rows


def kit_for_group(group: RoadUsageGroupRecipe, kits: dict[str, RoadKit]) -> RoadKit:
    """resolve the fitter kit for a usage group from group.kit"""
    return resolve_kit(group.kit, kits)


def flagged_rows(rows: Sequence[FitReportRow]) -> list[FitReportRow]:
    """rows that carry at least one flag"""
    return [row for row in rows if row.flags]


def format_fit_report(
    rows: Sequence[FitReportRow],
    *,
    merge_span: float,
) -> str:
    """plain-text fit report"""
    lines: list[str] = [
        f"road fit report  merge_span={merge_span:.2f}m  pieces={len(rows)}",
        (
            "poly  idx  role       stem                 "
            "x        z      yaw   gap_ahead  flags"
        ),
    ]
    row: FitReportRow
    for row in rows:
        gap_s: str
        if row.gap_ahead_m is None:
            gap_s = "   —"
        else:
            gap_s = f"{row.gap_ahead_m:7.2f}"
        flag_s: str = ",".join(row.flags) if row.flags else ""
        lines.append(
            f"{row.polyline_id[:12]:12s} {row.index:3d}  {row.role:10s} "
            f"{row.stem:20s} {row.x:8.1f} {row.z:8.1f} {row.yaw_deg:6.1f} "
            f"{gap_s}  {flag_s}"
        )
    bad: list[FitReportRow] = flagged_rows(rows)
    lines.append(f"flagged: {len(bad)}")
    bad_row: FitReportRow
    for bad_row in bad:
        lines.append(
            f"  [{bad_row.polyline_id} #{bad_row.index}] {bad_row.stem} "
            f"@ ({bad_row.x:.1f}, {bad_row.z:.1f}) "
            f"gap={bad_row.gap_ahead_m:.2f} flags={','.join(bad_row.flags)}"
        )
    return "\n".join(lines)


def write_fit_report(
    rows: Sequence[FitReportRow],
    *,
    merge_span: float,
    path: Path | None = None,
) -> Path:
    """overwrite the fit report file with a fresh report; return the path"""
    out: Path = default_fit_report_path() if path is None else path
    out.parent.mkdir(parents=True, exist_ok=True)
    text: str = format_fit_report(rows, merge_span=merge_span)
    out.write_text(text + "\n", encoding="utf-8")
    return out
