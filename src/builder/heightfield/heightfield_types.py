"""heightfield recipe types"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from math import gcd

from builder.generators.param_types import EnumParam, FloatParam
from builder.heightfield.heightmap_catalog import HeightmapId

# ushort indices: tile_verts^2 must stay <= 65535
heightfield_tile_step: int = 128
heightfield_tile_verts: int = heightfield_tile_step + 1
# world size snaps to multiples of 256m (and to a spacing-compatible quantum)
heightfield_size_multiple: int = 256
# 32k m at 1 m spacing → 32768 / 128 = 256 tiles
heightfield_max_tiles_per_axis: int = 256


class VertexSpacing(IntEnum):
    """metres between adjacent heightfield vertices"""

    m1 = 1
    m2 = 2
    m4 = 4


@dataclass(frozen=True)
class HeightmapLayer:
    """one heightmap contributing to sampled height (signed float32 sample space)"""

    heightmap_id: HeightmapId
    amplitude: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class Heightfield:
    """tiled heightfield recipe on a group node

    author size (metres) and vertex spacing; tiles are derived
    """

    size_x: FloatParam = FloatParam(
        256.0,
        label="Size X",
        step=float(heightfield_size_multiple),
        minimum=float(heightfield_size_multiple),
    )
    size_z: FloatParam = FloatParam(
        256.0,
        label="Size Z",
        step=float(heightfield_size_multiple),
        minimum=float(heightfield_size_multiple),
    )
    spacing: EnumParam = EnumParam(VertexSpacing.m1, label="Spacing")
    layers: tuple[HeightmapLayer, ...] = ()


@dataclass(frozen=True)
class HeightfieldLayout:
    """derived tile grid for a normalized heightfield"""

    tiles_x: int
    tiles_z: int
    tile_verts: int
    res_x: int
    res_z: int
    spacing: float
    size_x: float
    size_z: float


def spacing_metres(heightfield: Heightfield) -> float:
    """vertex spacing in metres from the spacing enum"""
    return float(heightfield.spacing.value)


def size_snap_quantum(spacing: float) -> float:
    """smallest size step that is a multiple of 256 and tiles with step 128"""
    spacing_i: int = int(round(spacing))
    if spacing_i < 1:
        raise ValueError("spacing must be at least 1 metre")
    cell: int = heightfield_tile_step * spacing_i
    # lcm(256, cell) so size/spacing is divisible by tile_step
    return float(heightfield_size_multiple * cell // gcd(heightfield_size_multiple, cell))


def snap_heightfield_size(metres: float, spacing: float) -> float:
    """snap world size up/down to a valid quantum for this spacing"""
    quantum: float = size_snap_quantum(spacing)
    if metres < quantum:
        return quantum
    return round(metres / quantum) * quantum


def normalize_heightfield(heightfield: Heightfield) -> Heightfield:
    """snap sizes to valid quanta for the current spacing"""
    spacing: float = spacing_metres(heightfield)
    size_x: float = snap_heightfield_size(heightfield.size_x.value, spacing)
    size_z: float = snap_heightfield_size(heightfield.size_z.value, spacing)
    return replace(
        heightfield,
        size_x=replace(heightfield.size_x, value=size_x),
        size_z=replace(heightfield.size_z, value=size_z),
    )


def heightfield_layout(heightfield: Heightfield) -> HeightfieldLayout:
    """derive tile counts and resolution from size and spacing"""
    spacing: float = spacing_metres(heightfield)
    size_x: float = snap_heightfield_size(heightfield.size_x.value, spacing)
    size_z: float = snap_heightfield_size(heightfield.size_z.value, spacing)
    intervals_x: int = int(round(size_x / spacing))
    intervals_z: int = int(round(size_z / spacing))
    if intervals_x % heightfield_tile_step != 0 or intervals_z % heightfield_tile_step != 0:
        raise ValueError("heightfield size/spacing does not tile with step 128")
    tiles_x: int = intervals_x // heightfield_tile_step
    tiles_z: int = intervals_z // heightfield_tile_step
    if tiles_x < 1 or tiles_z < 1:
        raise ValueError("heightfield needs at least one tile per axis")
    if tiles_x > heightfield_max_tiles_per_axis or tiles_z > heightfield_max_tiles_per_axis:
        raise ValueError(
            f"heightfield exceeds {heightfield_max_tiles_per_axis} tiles per axis"
        )
    return HeightfieldLayout(
        tiles_x=tiles_x,
        tiles_z=tiles_z,
        tile_verts=heightfield_tile_verts,
        res_x=intervals_x + 1,
        res_z=intervals_z + 1,
        spacing=spacing,
        size_x=size_x,
        size_z=size_z,
    )


def heightfield_resolution(heightfield: Heightfield) -> tuple[int, int]:
    """return full-field vertex counts (x, z)"""
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    return (layout.res_x, layout.res_z)


def default_heightfield() -> Heightfield:
    """return a small flat heightfield for placement"""
    return normalize_heightfield(Heightfield())
