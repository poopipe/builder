"""heightfield recipe types"""

from __future__ import annotations

from dataclasses import dataclass

from builder.generators.param_types import FloatParam, IntParam
from builder.heightfield.heightmap_catalog import HeightmapId

# ushort mesh indices limit verts to 65535; stay under that per tile
heightfield_max_tile_verts: int = 256
heightfield_max_tiles_per_axis: int = 16


@dataclass(frozen=True)
class HeightmapLayer:
    """one heightmap contributing to sampled height"""

    heightmap_id: HeightmapId
    amplitude: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class Heightfield:
    """tiled heightfield recipe on a group node

    vertex resolution is tiles * (tile_verts - 1) + 1 on each axis
    """

    tiles_x: IntParam = IntParam(
        2, label="Tiles X", minimum=1, maximum=heightfield_max_tiles_per_axis
    )
    tiles_z: IntParam = IntParam(
        2, label="Tiles Z", minimum=1, maximum=heightfield_max_tiles_per_axis
    )
    size_x: FloatParam = FloatParam(16.0, label="Size X", step=0.5, minimum=0.01)
    size_z: FloatParam = FloatParam(16.0, label="Size Z", step=0.5, minimum=0.01)
    tile_verts: IntParam = IntParam(
        33,
        label="Tile verts",
        step=1,
        minimum=2,
        maximum=heightfield_max_tile_verts,
    )
    layers: tuple[HeightmapLayer, ...] = ()


def heightfield_resolution(heightfield: Heightfield) -> tuple[int, int]:
    """return full-field vertex counts (x, z)"""
    step: int = heightfield.tile_verts.value - 1
    return (
        heightfield.tiles_x.value * step + 1,
        heightfield.tiles_z.value * step + 1,
    )


def default_heightfield() -> Heightfield:
    """return a small flat heightfield for placement"""
    return Heightfield()
