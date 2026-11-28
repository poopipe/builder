"""sea plane tile visibility for heightfield-local y = 0"""

from __future__ import annotations

from builder.heightfield.heightfield_types import Heightfield, HeightfieldLayout
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height
from builder.heightfield.vector.vector_types import world_xz_to_uv

sea_level_y: float = 0.0
# inclusive samples per axis over each tile footprint
sea_tile_sample_count: int = 4


def sea_tile_visible(min_height: float) -> bool:
    """true when the sea section is not fully covered by terrain"""
    return min_height < sea_level_y


def tile_min_height(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layout: HeightfieldLayout,
    tile_x: int,
    tile_z: int,
) -> float:
    """approximate min terrain height over one tile via a coarse uv grid"""
    tile_world: float = float(layout.tile_verts - 1) * layout.spacing
    origin_x: float = -0.5 * layout.size_x
    origin_z: float = -0.5 * layout.size_z
    x0: float = origin_x + float(tile_x) * tile_world
    z0: float = origin_z + float(tile_z) * tile_world
    samples: int = sea_tile_sample_count
    if samples < 2:
        raise ValueError("sea_tile_sample_count must be at least 2")
    denom: float = float(samples - 1)
    min_h: float = float("inf")
    iz: int
    ix: int
    for iz in range(samples):
        z: float = z0 + (float(iz) / denom) * tile_world
        for ix in range(samples):
            x: float = x0 + (float(ix) / denom) * tile_world
            u: float
            v: float
            u, v = world_xz_to_uv(x, z, layout.size_x, layout.size_z)
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            h: float = sample_height(catalog, heightfield, u, v)
            if h < min_h:
                min_h = h
    if min_h == float("inf"):
        return sea_level_y
    return min_h


def build_sea_visibility(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layout: HeightfieldLayout,
) -> list[bool]:
    """per-tile sea visibility, row-major tile_z then tile_x"""
    visible: list[bool] = []
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        for tile_x in range(layout.tiles_x):
            min_h: float = tile_min_height(
                catalog, heightfield, layout, tile_x, tile_z
            )
            visible.append(sea_tile_visible(min_h))
    return visible


def tile_index_range_for_aabb(
    layout: HeightfieldLayout,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> tuple[int, int, int, int] | None:
    """inclusive tile index range overlapping a world xz aabb; None if empty"""
    if layout.tiles_x < 1 or layout.tiles_z < 1:
        return None
    tile_world: float = float(layout.tile_verts - 1) * layout.spacing
    if tile_world <= 1e-8:
        return None
    origin_x: float = -0.5 * layout.size_x
    origin_z: float = -0.5 * layout.size_z
    tx0: int = max(0, int((min_x - origin_x) / tile_world))
    tx1: int = min(layout.tiles_x - 1, int((max_x - origin_x) / tile_world))
    tz0: int = max(0, int((min_z - origin_z) / tile_world))
    tz1: int = min(layout.tiles_z - 1, int((max_z - origin_z) / tile_world))
    if tx0 > tx1 or tz0 > tz1:
        return None
    return (tx0, tx1, tz0, tz1)


def refresh_sea_visibility(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layout: HeightfieldLayout,
    visibility: list[bool],
    dirty_world_aabb: tuple[float, float, float, float] | None,
) -> list[bool]:
    """rebuild sea flags; when dirty_world_aabb is set only retile that band"""
    expected: int = layout.tiles_x * layout.tiles_z
    if dirty_world_aabb is None or len(visibility) != expected:
        return build_sea_visibility(catalog, heightfield, layout)
    tile_range: tuple[int, int, int, int] | None = tile_index_range_for_aabb(
        layout,
        dirty_world_aabb[0],
        dirty_world_aabb[1],
        dirty_world_aabb[2],
        dirty_world_aabb[3],
    )
    if tile_range is None:
        return visibility
    tx0: int
    tx1: int
    tz0: int
    tz1: int
    tx0, tx1, tz0, tz1 = tile_range
    tile_z: int
    tile_x: int
    for tile_z in range(tz0, tz1 + 1):
        for tile_x in range(tx0, tx1 + 1):
            index: int = tile_z * layout.tiles_x + tile_x
            min_h: float = tile_min_height(
                catalog, heightfield, layout, tile_x, tile_z
            )
            visibility[index] = sea_tile_visible(min_h)
    return visibility
