"""shared flat tile grids for gpu heightfield displace (per lod + edge mask)"""

from __future__ import annotations

from typing import Any

from pyray import (
    Material,
    Mesh,
    ffi,
    get_mesh_bounding_box,
    upload_mesh,
)

from builder.heightfield.gpu.lod import (
    edge_east,
    edge_north,
    edge_south,
    edge_west,
    heightfield_lod_max,
    lod_tile_step,
    lod_tile_verts,
)
from builder.heightfield.heightfield_types import heightfield_tile_step
from builder.view.prepare_mesh import PreparedMesh, release_prepared_mesh


def build_flat_grid_vertices(
    tile_verts: int, tile_world_size: float
) -> tuple[list[float], list[float], list[float], list[int]]:
    """positions, normals, texcoords, colors for a flat xz grid"""
    if tile_verts < 2:
        raise ValueError("tile_verts must be at least 2")
    if tile_verts * tile_verts > 65535:
        raise ValueError("tile exceeds unsigned-short index limit")
    step: int = tile_verts - 1
    cell: float = tile_world_size / float(step)
    positions: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    colors: list[int] = []
    local_z: int
    local_x: int
    for local_z in range(tile_verts):
        for local_x in range(tile_verts):
            positions.extend(
                [float(local_x) * cell, 0.0, float(local_z) * cell]
            )
            normals.extend([0.0, 1.0, 0.0])
            texcoords.extend(
                [
                    float(local_x) / float(step),
                    float(local_z) / float(step),
                ]
            )
            colors.extend([255, 255, 255, 255])
    return positions, normals, texcoords, colors


def vertex_active_on_edges(
    x: int, z: int, step: int, edge_mask: int
) -> bool:
    """false for odd edge verts that must match a coarser neighbor"""
    if z == 0 and (edge_mask & edge_south) and (x % 2 == 1):
        return False
    if z == step and (edge_mask & edge_north) and (x % 2 == 1):
        return False
    if x == 0 and (edge_mask & edge_west) and (z % 2 == 1):
        return False
    if x == step and (edge_mask & edge_east) and (z % 2 == 1):
        return False
    return True


def cell_fully_active(
    local_x: int, local_z: int, step: int, edge_mask: int
) -> bool:
    """true when all four cell corners survive edge collapse"""
    return (
        vertex_active_on_edges(local_x, local_z, step, edge_mask)
        and vertex_active_on_edges(local_x + 1, local_z, step, edge_mask)
        and vertex_active_on_edges(local_x + 1, local_z + 1, step, edge_mask)
        and vertex_active_on_edges(local_x, local_z + 1, step, edge_mask)
    )


def build_stitched_indices(tile_verts: int, edge_mask: int) -> list[int]:
    """triangle indices; collapsed edges use even-span fans only"""
    step: int = tile_verts - 1
    if step < 1:
        raise ValueError("tile_verts must be at least 2")
    if step > 1 and (step & (step - 1)) != 0:
        raise ValueError("tile step must be a power of two for lod stitching")

    def vid(x: int, z: int) -> int:
        return z * tile_verts + x

    indices: list[int] = []

    def add_tri(a: int, b: int, c: int) -> None:
        indices.extend([a, b, c])

    local_z: int
    local_x: int
    for local_z in range(step):
        for local_x in range(step):
            if not cell_fully_active(local_x, local_z, step, edge_mask):
                continue
            # i0=(x,z), i1=(x+1,z), i2=(x,z+1), i3=(x+1,z+1)
            i0: int = vid(local_x, local_z)
            i1: int = vid(local_x + 1, local_z)
            i2: int = vid(local_x, local_z + 1)
            i3: int = vid(local_x + 1, local_z + 1)
            add_tri(i0, i2, i1)
            add_tri(i1, i2, i3)

    # even-span edge strips: stitch fan + two side tris (covers two cells)
    mid: int
    if edge_mask & edge_west:
        for mid in range(1, step, 2):
            a: int = vid(0, mid - 1)
            b: int = vid(0, mid + 1)
            c: int = vid(1, mid - 1)
            d: int = vid(1, mid)
            e: int = vid(1, mid + 1)
            add_tri(a, b, d)
            add_tri(a, d, c)
            add_tri(b, e, d)
    if edge_mask & edge_east:
        for mid in range(1, step, 2):
            a = vid(step, mid - 1)
            b = vid(step, mid + 1)
            c = vid(step - 1, mid - 1)
            d = vid(step - 1, mid)
            e = vid(step - 1, mid + 1)
            add_tri(b, a, d)
            add_tri(a, c, d)
            add_tri(b, d, e)
    if edge_mask & edge_south:
        for mid in range(1, step, 2):
            a = vid(mid - 1, 0)
            b = vid(mid + 1, 0)
            c = vid(mid - 1, 1)
            d = vid(mid, 1)
            e = vid(mid + 1, 1)
            add_tri(b, a, d)
            add_tri(a, c, d)
            add_tri(b, d, e)
    if edge_mask & edge_north:
        for mid in range(1, step, 2):
            a = vid(mid - 1, step)
            b = vid(mid + 1, step)
            c = vid(mid - 1, step - 1)
            d = vid(mid, step - 1)
            e = vid(mid + 1, step - 1)
            add_tri(a, b, d)
            add_tri(a, d, c)
            add_tri(b, e, d)
    return indices


def build_flat_grid_mesh(
    tile_verts: int, tile_world_size: float, edge_mask: int = 0
) -> tuple[Mesh, tuple[Any, ...]]:
    """build a flat xz grid spanning [0, tile_world_size]; y = 0, uv = 0..1"""
    positions: list[float]
    normals: list[float]
    texcoords: list[float]
    colors: list[int]
    positions, normals, texcoords, colors = build_flat_grid_vertices(
        tile_verts, tile_world_size
    )
    indices: list[int] = build_stitched_indices(tile_verts, edge_mask)
    if not indices:
        raise ValueError("grid has no triangles")
    vertex_count: int = tile_verts * tile_verts
    triangle_count: int = len(indices) // 3
    vertices_buf: Any = ffi.new("float[]", positions)
    normals_buf: Any = ffi.new("float[]", normals)
    texcoords_buf: Any = ffi.new("float[]", texcoords)
    colors_buf: Any = ffi.new("unsigned char[]", colors)
    indices_buf: Any = ffi.new("unsigned short[]", indices)
    mesh_cdata: Any = ffi.new("Mesh *")
    mesh_cdata.vertexCount = vertex_count
    mesh_cdata.triangleCount = triangle_count
    mesh_cdata.vertices = vertices_buf
    mesh_cdata.normals = normals_buf
    mesh_cdata.texcoords = texcoords_buf
    mesh_cdata.colors = colors_buf
    mesh_cdata.indices = indices_buf
    mesh: Mesh = mesh_cdata[0]
    upload_mesh(mesh, False)
    keep_alive: tuple[Any, ...] = (
        vertices_buf,
        normals_buf,
        texcoords_buf,
        colors_buf,
        indices_buf,
        mesh_cdata,
    )
    return mesh, keep_alive


def make_lod_grid_prepared(
    material: Material, spacing: float, lod: int, edge_mask: int
) -> PreparedMesh:
    """upload one tile grid; reuses the shared terrain material"""
    step: int = lod_tile_step(lod)
    tile_verts: int = lod_tile_verts(lod)
    tile_world_size: float = float(heightfield_tile_step) * spacing
    # coarser lods still span the full tile; verify step matches verts
    if step != tile_verts - 1:
        raise RuntimeError("lod step/verts mismatch")
    mesh: Mesh
    keep_alive: tuple[Any, ...]
    mesh, keep_alive = build_flat_grid_mesh(
        tile_verts, tile_world_size, edge_mask
    )
    return PreparedMesh(
        mesh=mesh,
        material=material,
        local_bounds=get_mesh_bounding_box(mesh),
        keep_alive=keep_alive,
    )


def make_tile_grid_prepared(material: Material, spacing: float) -> PreparedMesh:
    """upload lod-0 full-density grid (no edge collapse)"""
    return make_lod_grid_prepared(material, spacing, 0, 0)


def ensure_lod_grid(
    cache: dict[tuple[float, int, int], PreparedMesh],
    material: Material,
    spacing: float,
    lod: int,
    edge_mask: int,
) -> PreparedMesh:
    """return a cached grid for (spacing, lod, edge_mask), uploading once"""
    key: tuple[float, int, int] = (spacing, lod, edge_mask)
    existing: PreparedMesh | None = cache.get(key)
    if existing is not None:
        return existing
    prepared: PreparedMesh = make_lod_grid_prepared(
        material, spacing, lod, edge_mask
    )
    cache[key] = prepared
    return prepared


def prewarm_lod_grids(
    cache: dict[tuple[float, int, int], PreparedMesh],
    material: Material,
    spacing: float,
) -> None:
    """upload all lod × edge-mask variants for spacing up front"""
    lod: int
    edge_mask: int
    for lod in range(heightfield_lod_max + 1):
        for edge_mask in range(16):
            ensure_lod_grid(cache, material, spacing, lod, edge_mask)


def release_tile_grid(prepared: PreparedMesh | None) -> None:
    """unload a shared tile grid if present"""
    if prepared is None:
        return
    release_prepared_mesh(prepared)


def release_all_lod_grids(
    cache: dict[tuple[float, int, int], PreparedMesh],
) -> None:
    """unload every cached lod grid"""
    key: tuple[float, int, int]
    for key in list(cache):
        release_tile_grid(cache.pop(key))
