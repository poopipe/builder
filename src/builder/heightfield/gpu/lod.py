"""progressive half-density lod selection and edge masks"""

from __future__ import annotations

from builder.heightfield.heightfield_types import (
    HeightfieldLayout,
    heightfield_tile_step,
)

# lod 0 = full density (129 verts); lod 5 = step/32 + 1 (5 verts)
heightfield_lod_max: int = 5

# |dy| scale for hybrid distance: max(xz, scale * |dy|)
lod_distance_vertical_scale: float = 1.0

# edge bit: neighbor on that side is coarser (collapse odd verts on that edge)
edge_north: int = 1
edge_east: int = 2
edge_south: int = 4
edge_west: int = 8


def hybrid_lod_distance(
    dx: float, dy: float, dz: float, *, vertical_scale: float = lod_distance_vertical_scale
) -> float:
    """effective lod distance: max(xz, vertical_scale * |dy|)"""
    distance_xz: float = (dx * dx + dz * dz) ** 0.5
    distance_y: float = abs(dy) * vertical_scale
    if distance_y > distance_xz:
        return distance_y
    return distance_xz


def lod_tile_step(lod: int) -> int:
    """vertex intervals along one tile edge at this lod"""
    if lod < 0 or lod > heightfield_lod_max:
        raise ValueError(f"lod must be 0..{heightfield_lod_max}")
    return heightfield_tile_step >> lod


def lod_tile_verts(lod: int) -> int:
    """vertex count along one tile edge at this lod"""
    return lod_tile_step(lod) + 1


def raw_lod_from_distance(distance: float, tile_world_size: float) -> int:
    """map camera distance to a raw lod using tile-size thresholds"""
    if tile_world_size <= 0.0:
        return heightfield_lod_max
    units: float = distance / tile_world_size
    if units < 2.0:
        return 0
    if units < 4.0:
        return 1
    if units < 8.0:
        return 2
    if units < 16.0:
        return 3
    if units < 32.0:
        return 4
    return 5


def clamp_neighbor_lods(levels: list[list[int]], tiles_x: int, tiles_z: int) -> None:
    """enforce |lod - neighbor| <= 1 in place (iterate to a fixed point)"""
    changed: bool = True
    while changed:
        changed = False
        tile_z: int
        tile_x: int
        for tile_z in range(tiles_z):
            for tile_x in range(tiles_x):
                level: int = levels[tile_z][tile_x]
                pairs: list[tuple[int, int]] = []
                if tile_x > 0:
                    pairs.append((tile_x - 1, tile_z))
                if tile_x + 1 < tiles_x:
                    pairs.append((tile_x + 1, tile_z))
                if tile_z > 0:
                    pairs.append((tile_x, tile_z - 1))
                if tile_z + 1 < tiles_z:
                    pairs.append((tile_x, tile_z + 1))
                nx: int
                nz: int
                for nx, nz in pairs:
                    neighbor: int = levels[nz][nx]
                    if level > neighbor + 1:
                        levels[tile_z][tile_x] = neighbor + 1
                        level = neighbor + 1
                        changed = True
                    elif neighbor > level + 1:
                        levels[nz][nx] = level + 1
                        changed = True


def edge_mask_for_tile(
    levels: list[list[int]], tile_x: int, tile_z: int, tiles_x: int, tiles_z: int
) -> int:
    """bits for edges whose neighbor is coarser than this tile"""
    level: int = levels[tile_z][tile_x]
    mask: int = 0
    if tile_z + 1 < tiles_z and levels[tile_z + 1][tile_x] > level:
        mask |= edge_north
    if tile_x + 1 < tiles_x and levels[tile_z][tile_x + 1] > level:
        mask |= edge_east
    if tile_z > 0 and levels[tile_z - 1][tile_x] > level:
        mask |= edge_south
    if tile_x > 0 and levels[tile_z][tile_x - 1] > level:
        mask |= edge_west
    return mask


def select_tile_lods(
    layout: HeightfieldLayout,
    camera_x: float,
    camera_y: float,
    camera_z: float,
    field_origin_x: float,
    field_origin_z: float,
    field_plane_y: float = 0.0,
) -> tuple[list[list[int]], list[list[int]]]:
    """return per-tile lod levels and edge masks (row-major z, then x)"""
    tile_world: float = float(layout.tile_verts - 1) * layout.spacing
    levels: list[list[int]] = []
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        row: list[int] = []
        for tile_x in range(layout.tiles_x):
            center_x: float = (
                field_origin_x
                + (float(tile_x) + 0.5) * tile_world
            )
            center_z: float = (
                field_origin_z
                + (float(tile_z) + 0.5) * tile_world
            )
            dx: float = center_x - camera_x
            dy: float = field_plane_y - camera_y
            dz: float = center_z - camera_z
            distance: float = hybrid_lod_distance(dx, dy, dz)
            row.append(raw_lod_from_distance(distance, tile_world))
        levels.append(row)
    clamp_neighbor_lods(levels, layout.tiles_x, layout.tiles_z)
    masks: list[list[int]] = []
    for tile_z in range(layout.tiles_z):
        mask_row: list[int] = []
        for tile_x in range(layout.tiles_x):
            mask_row.append(
                edge_mask_for_tile(
                    levels, tile_x, tile_z, layout.tiles_x, layout.tiles_z
                )
            )
        masks.append(mask_row)
    return levels, masks
