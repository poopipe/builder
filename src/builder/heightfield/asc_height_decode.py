"""ESRI ASCII grid (.asc) height decode and optional sidecar meta"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
)


@dataclass(frozen=True)
class AscGridHeader:
    """parsed ESRI ASCII grid header"""

    ncols: int
    nrows: int
    cellsize: float
    nodata: float | None
    # map-space lower-left (west, south); metres
    xllcorner: float = 0.0
    yllcorner: float = 0.0


@dataclass(frozen=True)
class AscHeightMeta:
    """optional elevation bounds from {stem}_meta.json"""

    height_min: float
    height_max: float


@dataclass(frozen=True)
class AscImportHints:
    """world size, map origin, and optional height bounds for base import"""

    size_x: float
    size_z: float
    origin_x: float
    origin_z: float
    height_min: float | None
    height_max: float | None


def asc_height_meta_path(asc_path: Path) -> Path:
    """sidecar path next to the .asc file"""
    return asc_path.parent / f"{asc_path.stem}_meta.json"


def load_asc_height_meta(asc_path: Path) -> AscHeightMeta | None:
    """read elevation_m min/max from optional sidecar; None if missing or invalid"""
    meta_path: Path = asc_height_meta_path(asc_path)
    if not meta_path.is_file():
        return None
    try:
        raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    elevation: object = raw.get("elevation_m")
    if not isinstance(elevation, dict):
        return None
    min_raw: object = elevation.get("min")
    max_raw: object = elevation.get("max")
    if not isinstance(min_raw, (int, float)) or not isinstance(max_raw, (int, float)):
        return None
    return AscHeightMeta(height_min=float(min_raw), height_max=float(max_raw))


def parse_asc_header(lines: list[str]) -> tuple[AscGridHeader, int]:
    """parse leading header lines; return header and index of first data line"""
    fields: dict[str, float] = {}
    index: int = 0
    while index < len(lines):
        line: str = lines[index].strip()
        if line == "":
            index += 1
            continue
        parts: list[str] = line.split()
        if len(parts) < 2:
            break
        key: str = parts[0].lower()
        if key not in (
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "xllcenter",
            "yllcenter",
            "cellsize",
            "nodata_value",
        ):
            break
        try:
            fields[key] = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid asc header value for {key}: {parts[1]!r}") from exc
        index += 1
    if "ncols" not in fields or "nrows" not in fields or "cellsize" not in fields:
        raise ValueError("asc header missing ncols, nrows, or cellsize")
    ncols: int = int(fields["ncols"])
    nrows: int = int(fields["nrows"])
    if ncols < 1 or nrows < 1:
        raise ValueError(f"asc grid has no cells: {ncols}x{nrows}")
    cellsize: float = float(fields["cellsize"])
    if cellsize <= 0.0:
        raise ValueError(f"asc cellsize must be positive, got {cellsize}")
    nodata: float | None = None
    if "nodata_value" in fields:
        nodata = float(fields["nodata_value"])
    xllcorner: float
    yllcorner: float
    if "xllcorner" in fields:
        xllcorner = float(fields["xllcorner"])
    elif "xllcenter" in fields:
        xllcorner = float(fields["xllcenter"]) - 0.5 * cellsize
    else:
        xllcorner = 0.0
    if "yllcorner" in fields:
        yllcorner = float(fields["yllcorner"])
    elif "yllcenter" in fields:
        yllcorner = float(fields["yllcenter"]) - 0.5 * cellsize
    else:
        yllcorner = 0.0
    return (
        AscGridHeader(
            ncols=ncols,
            nrows=nrows,
            cellsize=cellsize,
            nodata=nodata,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
        ),
        index,
    )


def read_asc_header(path: Path) -> AscGridHeader:
    """read only the ASCII grid header from path"""
    text: str = path.read_text(encoding="utf-8")
    lines: list[str] = text.splitlines()
    header, _ = parse_asc_header(lines)
    return header


def pack_f32_le(values: list[float]) -> bytes:
    """pack float32 samples as little-endian bytes"""
    return struct.pack(f"<{len(values)}f", *values)


def asc_decode_height_image(path: Path) -> DecodedHeightImage:
    """decode ESRI .asc to f32 metres; south-first rows, east-west mirrored columns"""
    text: str = path.read_text(encoding="utf-8")
    lines: list[str] = text.splitlines()
    header, data_index = parse_asc_header(lines)
    ncols: int = header.ncols
    nrows: int = header.nrows
    expected: int = ncols * nrows
    tokens: list[str] = []
    line: str
    for line in lines[data_index:]:
        stripped: str = line.strip()
        if stripped == "":
            continue
        tokens.extend(stripped.split())
    if len(tokens) != expected:
        raise ValueError(
            f"asc expected {expected} samples, got {len(tokens)}: {path}"
        )
    samples_north_first: list[float] = []
    token: str
    for token in tokens:
        try:
            value: float = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid asc sample {token!r} in {path}") from exc
        if header.nodata is not None and value == header.nodata:
            samples_north_first.append(0.0)
        else:
            samples_north_first.append(value)
    # ASC: row 0 = north, col 0 = west; buffer: row 0 = south, col 0 = east
    samples: list[float] = []
    row: int
    for row in range(nrows - 1, -1, -1):
        start: int = row * ncols
        row_vals: list[float] = samples_north_first[start : start + ncols]
        samples.extend(reversed(row_vals))
    return DecodedHeightImage(
        width=ncols,
        height=nrows,
        sample_type=HeightSampleType.f32,
        data=pack_f32_le(samples),
    )


def asc_import_hints(asc_path: Path) -> AscImportHints:
    """world size from cellsize and optional meta elevation bounds"""
    header: AscGridHeader = read_asc_header(asc_path)
    meta: AscHeightMeta | None = load_asc_height_meta(asc_path)
    height_min: float | None = None
    height_max: float | None = None
    if meta is not None:
        height_min = meta.height_min
        height_max = meta.height_max
    return AscImportHints(
        size_x=float(header.ncols) * header.cellsize,
        size_z=float(header.nrows) * header.cellsize,
        origin_x=header.xllcorner,
        origin_z=header.yllcorner,
        height_min=height_min,
        height_max=height_max,
    )
