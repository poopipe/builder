"""dayz road prefab kits discovered from on-disk parts folders"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from math import radians
from pathlib import Path

from builder.heightfield.road_sections.kit_widths import PieceSnapPoints

default_parts_roots: tuple[Path, ...] = (
    Path(r"P:\DZ\structures_bliss\roads"),
    Path(r"P:\DZ\structures\roads"),
    Path(r"P:\DZ\structures_sakhal\roads"),
    Path(r"P:\DZ\structures_takistan\Roads"),
)

straight_lengths: frozenset[int] = frozenset({6, 10, 12, 25, 30})
known_radii: tuple[int, ...] = (2000, 1000, 100, 75, 50, 25, 10)
curve_angle_deg: float = 30.0
curve_radius_m: float = 38.0

re_corner_spaced: re.Pattern[str] = re.compile(r"^(.+)_(\d+) (\d+)$")
re_konec: re.Pattern[str] = re.compile(r"^(.+)_6konec$")
re_end: re.Pattern[str] = re.compile(r"^(.+)_6end\d*$")
re_curve: re.Pattern[str] = re.compile(r"^(.+)_curve$")
re_digits: re.Pattern[str] = re.compile(r"^(.+)_(\d+)$")

# kit used when usage has no explicit mapping
default_usage_kit: str = "asf3"

# usage class -> dayz kit name
usage_to_kit: dict[str, str] = {
    "provincial": "asf1",
    "primary": "asf1",
    "trunk": "asf1",
    "motorway": "asf1",
    "secondary": "asf2",
    "tertiary": "asf3",
    "residential": "city",
    "living_street": "city",
    "unclassified": "asf3",
    "service": "grav",
    "track": "mud",
    "trackunclassified": "mud",
    "softtrack": "mud",
    "dirtroad": "grav",
    "footway": "path_dirt",
    "path": "path_dirt",
    "pedestrian": "path_dirt",
    "cycleway": "path_dirt",
    "bridleway": "path_dirt",
    "steps": "path_rock",
}


@dataclass(frozen=True)
class CornerPiece:
    """corner prefab: angle degrees and radius metres"""

    angle_deg: float
    radius_m: float
    stem: str

    @property
    def arc_length_m(self) -> float:
        return self.radius_m * radians(self.angle_deg)


@dataclass(frozen=True)
class RoadKit:
    """one kit's straights, corners, and optional terminator"""

    name: str
    straights_m: tuple[int, ...]
    terminator: str | None
    corners: tuple[CornerPiece, ...] = ()


class PieceType(IntEnum):
    straight = 1
    corner = 2
    terminator = 3
    curve = 4


@dataclass(frozen=True)
class ParsedStem:
    """classified .p3d stem"""

    piece_type: PieceType
    kit: str
    stem: str
    length_m: int | None = None
    angle_deg: float | None = None
    radius_m: float | None = None
    is_konec: bool = False


def parts_dir(root: Path) -> Path | None:
    """resolve a parts directory from a roads root or a parts path itself"""
    root = Path(root)
    if not root.exists():
        return None
    if root.is_dir() and root.name.lower() == "parts":
        return root
    if not root.is_dir():
        return None
    child: Path
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == "parts":
            return child
    return None


def split_takistan_corner(digits: str) -> tuple[float, float] | None:
    """split concatenated {angle}{radius} using known radii suffixes"""
    radius: int
    for radius in known_radii:
        suffix: str = str(radius)
        if not digits.endswith(suffix):
            continue
        angle_s: str = digits[: -len(suffix)]
        if not angle_s or not angle_s.isdigit():
            continue
        return float(int(angle_s)), float(radius)
    return None


def parse_stem(stem: str) -> ParsedStem | None:
    """classify a .p3d stem into a road-fitter piece, or none to ignore"""
    lower: str = stem.lower()
    if stem.startswith("kr_") or stem.startswith("kr "):
        return None
    if "crosswalk" in lower or "_merge_" in lower or "_exit_" in lower:
        return None

    match = re_corner_spaced.match(stem)
    if match is not None:
        kit: str = match.group(1)
        angle: float = float(match.group(2))
        if angle <= 1.0:
            return None
        return ParsedStem(
            PieceType.corner,
            kit,
            stem,
            angle_deg=angle,
            radius_m=float(match.group(3)),
        )

    match = re_konec.match(stem)
    if match is not None:
        return ParsedStem(
            PieceType.terminator, match.group(1), stem, is_konec=True
        )

    match = re_end.match(stem)
    if match is not None:
        return ParsedStem(
            PieceType.terminator, match.group(1), stem, is_konec=False
        )

    match = re_curve.match(stem)
    if match is not None:
        return ParsedStem(
            PieceType.curve,
            match.group(1),
            stem,
            angle_deg=curve_angle_deg,
            radius_m=curve_radius_m,
        )

    match = re_digits.match(stem)
    if match is not None:
        kit = match.group(1)
        num_s: str = match.group(2)
        n: int = int(num_s)
        if n in straight_lengths:
            return ParsedStem(PieceType.straight, kit, stem, length_m=n)
        split: tuple[float, float] | None = split_takistan_corner(num_s)
        if split is not None:
            angle, radius = split
            if angle <= 1.0:
                return None
            return ParsedStem(
                PieceType.corner,
                kit,
                stem,
                angle_deg=angle,
                radius_m=radius,
            )
        return None

    return None


@dataclass
class KitAccum:
    """mutable accumulator while scanning parts"""

    straights: dict[int, str] = field(default_factory=dict)
    corners: dict[str, CornerPiece] = field(default_factory=dict)
    terminator_konec: str | None = None
    terminator_end: str | None = None


def build_kits(roots: Sequence[Path] | None = None) -> dict[str, RoadKit]:
    """scan parts/*.p3d under each root and build fitter kits"""
    use_roots: tuple[Path, ...] = (
        default_parts_roots if roots is None else tuple(Path(r) for r in roots)
    )
    seen_stems: set[str] = set()
    acc: dict[str, KitAccum] = {}

    root: Path
    for root in use_roots:
        parts: Path | None = parts_dir(root)
        if parts is None:
            continue
        p3d: Path
        for p3d in sorted(parts.glob("*.p3d")):
            stem: str = p3d.stem
            if stem in seen_stems:
                continue
            parsed: ParsedStem | None = parse_stem(stem)
            if parsed is None:
                continue
            seen_stems.add(stem)
            bucket: KitAccum = acc.setdefault(parsed.kit, KitAccum())
            if (
                parsed.piece_type is PieceType.straight
                and parsed.length_m is not None
            ):
                bucket.straights.setdefault(parsed.length_m, stem)
            elif parsed.piece_type in (PieceType.corner, PieceType.curve):
                assert (
                    parsed.angle_deg is not None and parsed.radius_m is not None
                )
                bucket.corners.setdefault(
                    stem,
                    CornerPiece(parsed.angle_deg, parsed.radius_m, stem),
                )
            elif parsed.piece_type is PieceType.terminator:
                if parsed.is_konec:
                    if bucket.terminator_konec is None:
                        bucket.terminator_konec = stem
                elif bucket.terminator_end is None:
                    bucket.terminator_end = stem

    kits: dict[str, RoadKit] = {}
    name: str
    bucket: KitAccum
    for name, bucket in acc.items():
        if not bucket.straights:
            continue
        lengths: tuple[int, ...] = tuple(
            sorted(bucket.straights.keys(), reverse=True)
        )
        corners: tuple[CornerPiece, ...] = tuple(
            sorted(
                bucket.corners.values(),
                key=lambda c: (-c.angle_deg, c.radius_m),
            )
        )
        terminator: str | None = (
            bucket.terminator_konec or bucket.terminator_end
        )
        kits[name] = RoadKit(name, lengths, terminator, corners)
    return kits


def straight_stem(kit: RoadKit, length_m: int) -> str:
    """stem name for a straight of the given length"""
    return f"{kit.name}_{length_m}"


def kit_for_usage(usage: str) -> str | None:
    """explicit kit for a usage class, or none when unmapped"""
    return usage_to_kit.get(usage.strip().lower())


def require_kit_for_usage(usage: str) -> str:
    """default kit for a usage class; falls back to asf3 when unmapped"""
    kit_name: str | None = kit_for_usage(usage)
    if kit_name is None:
        return default_usage_kit
    return kit_name


def resolve_kit(kit_name: str, kits: dict[str, RoadKit]) -> RoadKit:
    """lookup a kit by name; raises when missing"""
    kit: RoadKit | None = kits.get(kit_name)
    if kit is None:
        raise ValueError(f"kit {kit_name!r} not found in parts catalog")
    return kit


@dataclass
class RoadPartsCatalog:
    """loaded kits plus stem → proxy size; owned by the application"""

    kits: dict[str, RoadKit] = field(default_factory=dict)
    stem_length_m: dict[str, float] = field(default_factory=dict)
    stem_corners: dict[str, CornerPiece] = field(default_factory=dict)
    kit_width_m: dict[str, float] = field(default_factory=dict)
    stem_snaps: dict[str, PieceSnapPoints] = field(default_factory=dict)
    loaded: bool = False

    def ensure_loaded(self, roots: Sequence[Path] | None = None) -> None:
        """scan parts roots once; missing roots are skipped"""
        if self.loaded:
            return
        from builder.heightfield.road_sections.kit_widths import (
            load_kit_widths,
            load_piece_metrics,
        )

        self.kits = build_kits(roots)
        lengths: dict[str, float] = {}
        corners: dict[str, CornerPiece] = {}
        kit: RoadKit
        for kit in self.kits.values():
            length: int
            for length in kit.straights_m:
                lengths[straight_stem(kit, length)] = float(length)
            corner: CornerPiece
            for corner in kit.corners:
                lengths[corner.stem] = float(corner.arc_length_m)
                corners[corner.stem] = corner
            if kit.terminator is not None:
                lengths.setdefault(kit.terminator, 6.0)
        snaps: dict[str, PieceSnapPoints] = {}
        try:
            self.kit_width_m = load_kit_widths()
            metrics = load_piece_metrics()
            stem: str
            for stem, row in metrics.items():
                if row.snap_points is not None:
                    snaps[stem] = row.snap_points
                measured: float | None = None
                if row.snap_points is not None:
                    measured = row.snap_points.length_m()
                elif row.length_axis_m is not None:
                    measured = row.length_axis_m
                if measured is not None and measured > 0.5:
                    # prefer measured packing length for straights / terminators
                    if stem in lengths and stem not in corners:
                        lengths[stem] = measured
        except FileNotFoundError:
            self.kit_width_m = {}
        self.stem_length_m = lengths
        self.stem_corners = corners
        self.stem_snaps = snaps
        self.loaded = True

    def length_for_stem(self, stem: str) -> float:
        """proxy / pack length for a stem; default 6 m when unknown"""
        self.ensure_loaded()
        return self.stem_length_m.get(stem, 6.0)

    def corner_for_stem(self, stem: str) -> CornerPiece | None:
        """corner geometry for a stem, or none for straights/terminators"""
        return self.stem_corners.get(stem)

    def snaps_for_stem(self, stem: str) -> PieceSnapPoints | None:
        """lb/le/pb/pe mating points for a stem, or none"""
        self.ensure_loaded()
        return self.stem_snaps.get(stem)

    def width_for_kit(self, kit_name: str) -> float:
        """full road width metres for a kit; 6 m when catalogue has no entry"""
        self.ensure_loaded()
        return float(self.kit_width_m.get(kit_name, 6.0))
