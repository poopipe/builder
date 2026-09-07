"""build a single heightfield tile mesh and upload it"""

from __future__ import annotations

from typing import Any

from pyray import (
    Color,
    MaterialMapIndex,
    Mesh,
    Shader,
    ffi,
    upload_mesh,
)

from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height
from builder.view.prepare_mesh import PreparedMesh, prepare_mesh

# terrain albedo under vertex colors (default material white texture)
heightfield_albedo: Color = Color(210, 210, 210, 255)


def lerp_byte(a: int, b: int, t: float) -> int:
    """lerp two 0..255 channels"""
    return int(round(float(a) + (float(b) - float(a)) * t))


def terrain_vertex_color(
    height: float,
    height_min: float,
    height_max: float,
    normal_y: float,
    global_x: int,
    global_z: int,
) -> tuple[int, int, int, int]:
    """height gradient + slope darkening + checker so relief reads clearly"""
    span: float = height_max - height_min
    t: float = 0.5 if span < 1e-6 else (height - height_min) / span
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    # low valley -> mid grass -> high rock
    if t < 0.45:
        u: float = t / 0.45
        r: int = lerp_byte(42, 78, u)
        g: int = lerp_byte(92, 140, u)
        b: int = lerp_byte(58, 72, u)
    else:
        u = (t - 0.45) / 0.55
        r = lerp_byte(78, 168, u)
        g = lerp_byte(140, 148, u)
        b = lerp_byte(72, 118, u)
    # flatten toward rock gray on steep slopes
    slope: float = 1.0 - max(0.0, min(1.0, normal_y))
    rock: float = min(1.0, slope * 1.35)
    r = lerp_byte(r, 118, rock)
    g = lerp_byte(g, 112, rock)
    b = lerp_byte(b, 108, rock)
    # checker by world cell so tiles share a continuous grid
    check: int = 4
    if ((global_x // check) + (global_z // check)) % 2 == 0:
        r = max(0, int(round(float(r) * 0.82)))
        g = max(0, int(round(float(g) * 0.82)))
        b = max(0, int(round(float(b) * 0.82)))
    return (r, g, b, 255)


def build_tile_heights(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    tile_x: int,
    tile_z: int,
) -> list[float]:
    """sample heights for one tile's vertex grid (row-major x then z)"""
    tile_verts: int = heightfield.tile_verts.value
    step: int = tile_verts - 1
    res_x: int
    res_z: int
    res_x = heightfield.tiles_x.value * step + 1
    res_z = heightfield.tiles_z.value * step + 1
    origin_i: int = tile_x * step
    origin_k: int = tile_z * step
    heights: list[float] = []
    local_z: int
    local_x: int
    for local_z in range(tile_verts):
        global_z: int = origin_k + local_z
        v: float = 0.0 if res_z <= 1 else float(global_z) / float(res_z - 1)
        for local_x in range(tile_verts):
            global_x: int = origin_i + local_x
            u: float = 0.0 if res_x <= 1 else float(global_x) / float(res_x - 1)
            heights.append(sample_height(catalog, heightfield.layers, u, v))
    return heights


def field_height_range(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
) -> tuple[float, float]:
    """min/max height across all tiles for consistent vertex coloring"""
    height_min: float = 0.0
    height_max: float = 0.0
    first: bool = True
    tile_z: int
    tile_x: int
    for tile_z in range(heightfield.tiles_z.value):
        for tile_x in range(heightfield.tiles_x.value):
            heights: list[float] = build_tile_heights(
                catalog, heightfield, tile_x, tile_z
            )
            if not heights:
                continue
            local_min: float = min(heights)
            local_max: float = max(heights)
            if first:
                height_min = local_min
                height_max = local_max
                first = False
            else:
                height_min = min(height_min, local_min)
                height_max = max(height_max, local_max)
    if first or abs(height_max - height_min) < 1e-6:
        return (height_min - 0.5, height_max + 0.5)
    return (height_min, height_max)


def upload_tile_mesh(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    tile_x: int,
    tile_z: int,
    shader: Shader,
    height_min: float,
    height_max: float,
) -> PreparedMesh:
    """create and upload one tile mesh in local tile space (origin at tile min xz)"""
    tile_verts: int = heightfield.tile_verts.value
    if tile_verts < 2:
        raise ValueError("tile_verts must be at least 2")
    if tile_verts * tile_verts > 65535:
        raise ValueError("tile exceeds unsigned-short index limit")
    step: int = tile_verts - 1
    size_x: float = heightfield.size_x.value
    size_z: float = heightfield.size_z.value
    res_x: int = heightfield.tiles_x.value * step + 1
    res_z: int = heightfield.tiles_z.value * step + 1
    cell_x: float = size_x / float(res_x - 1) if res_x > 1 else size_x
    cell_z: float = size_z / float(res_z - 1) if res_z > 1 else size_z
    origin_i: int = tile_x * step
    origin_k: int = tile_z * step
    heights: list[float] = build_tile_heights(catalog, heightfield, tile_x, tile_z)

    positions: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    colors: list[int] = []
    local_z: int
    local_x: int
    for local_z in range(tile_verts):
        for local_x in range(tile_verts):
            y: float = heights[local_z * tile_verts + local_x]
            positions.extend(
                [float(local_x) * cell_x, y, float(local_z) * cell_z]
            )
            left_x: int = max(local_x - 1, 0)
            right_x: int = min(local_x + 1, tile_verts - 1)
            down_z: int = max(local_z - 1, 0)
            up_z: int = min(local_z + 1, tile_verts - 1)
            h_l: float = heights[local_z * tile_verts + left_x]
            h_r: float = heights[local_z * tile_verts + right_x]
            h_d: float = heights[down_z * tile_verts + local_x]
            h_u: float = heights[up_z * tile_verts + local_x]
            span_x: float = cell_x * float(max(right_x - left_x, 1))
            span_z: float = cell_z * float(max(up_z - down_z, 1))
            nx: float = -(h_r - h_l) / span_x
            ny: float = 1.0
            nz: float = -(h_u - h_d) / span_z
            length: float = (nx * nx + ny * ny + nz * nz) ** 0.5
            if length > 1e-12:
                nx /= length
                ny /= length
                nz /= length
            normals.extend([nx, ny, nz])
            texcoords.extend(
                [
                    float(local_x) / float(tile_verts - 1),
                    float(local_z) / float(tile_verts - 1),
                ]
            )
            r: int
            g: int
            b: int
            a: int
            r, g, b, a = terrain_vertex_color(
                y,
                height_min,
                height_max,
                ny,
                origin_i + local_x,
                origin_k + local_z,
            )
            colors.extend([r, g, b, a])

    indices: list[int] = []
    for local_z in range(step):
        for local_x in range(step):
            i0: int = local_z * tile_verts + local_x
            i1: int = i0 + 1
            i2: int = i0 + tile_verts
            i3: int = i2 + 1
            indices.extend([i0, i2, i1, i1, i2, i3])

    vertex_count: int = tile_verts * tile_verts
    triangle_count: int = step * step * 2
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
    prepared: PreparedMesh = prepare_mesh(mesh, shader)
    prepared.material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = (
        heightfield_albedo
    )
    return PreparedMesh(
        mesh=prepared.mesh,
        material=prepared.material,
        local_bounds=prepared.local_bounds,
        keep_alive=(
            mesh_cdata,
            vertices_buf,
            normals_buf,
            texcoords_buf,
            colors_buf,
            indices_buf,
        ),
    )
