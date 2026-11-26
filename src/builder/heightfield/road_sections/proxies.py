"""on-demand proxy meshes for road piece stems"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any

from pyray import Mesh, Shader, Transform, Vector3, ffi, gen_mesh_cube, upload_mesh

from builder.heightfield.road_sections.part_catalog import (
    CornerPiece,
    RoadPartsCatalog,
)
from builder.scene.orientation import quaternion_from_yaw_pitch_roll
from builder.scene.scene_types import MeshId
from builder.view.prepare_mesh import PreparedMesh, prepare_mesh

road_proxy_height_m: float = 0.3
# segments along the corner arc (minimum; grows with angle)
road_proxy_corner_seg_min: int = 8
road_proxy_corner_seg_per_deg: float = 0.25


@dataclass(frozen=True)
class CornerProxyGeometry:
    """cpu-side corner ribbon before gpu upload"""

    positions: tuple[float, ...]
    normals: tuple[float, ...]
    texcoords: tuple[float, ...]
    colors: tuple[int, ...]
    indices: tuple[int, ...]
    segments: int

    @property
    def vertex_count(self) -> int:
        return (self.segments + 1) * 4

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


def road_proxy_mesh_id(stem: str, width_m: float) -> MeshId:
    """stable mesh id for a catalog stem proxy at a given width"""
    return MeshId(f"road_proxy:{stem}:w{width_m:.3f}")


def make_road_proxy_mesh(length_m: float, width_m: float) -> Mesh:
    """thin box; length along local +Z, width along +X"""
    length: float = max(length_m, 0.5)
    width: float = max(width_m, 0.5)
    return gen_mesh_cube(width, road_proxy_height_m, length)


def corner_segment_count(angle_deg: float) -> int:
    """how many arc steps to approximate a corner"""
    from_angle: int = int(abs(angle_deg) * road_proxy_corner_seg_per_deg + 0.5)
    return max(road_proxy_corner_seg_min, from_angle)


def build_corner_proxy_geometry(
    corner: CornerPiece, width_m: float
) -> CornerProxyGeometry:
    """cw arc ribbon; origin at centerline mid-arc; entry heading +Z

    pivot at arc midpoint. cw: yaw = entry heading. ccw: yaw-spin the same
    mesh so it runs reverse (see corner_placement_yaw_deg) — keeps +Y up.
    """
    angle_deg: float = max(float(corner.angle_deg), 1.0)
    width: float = max(width_m, 0.5)
    radius: float = max(float(corner.radius_m), width)
    half_w: float = width * 0.5
    half_h: float = road_proxy_height_m * 0.5
    inner_r: float = max(radius - half_w, 0.05)
    outer_r: float = radius + half_w
    segments: int = corner_segment_count(angle_deg)
    half_angle: float = radians(angle_deg * 0.5)
    mid_x: float = radius * (1.0 - cos(half_angle))
    mid_z: float = radius * sin(half_angle)

    positions: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    colors: list[int] = []
    indices: list[int] = []

    ring: int
    for ring in range(segments + 1):
        t: float = float(ring) / float(segments)
        theta: float = radians(angle_deg * t)
        c: float = cos(theta)
        s: float = sin(theta)
        for radial, y in (
            (inner_r, -half_h),
            (outer_r, -half_h),
            (inner_r, half_h),
            (outer_r, half_h),
        ):
            px: float = radius - radial * c - mid_x
            pz: float = radial * s - mid_z
            positions.extend([px, y, pz])
            normals.extend([0.0, 1.0 if y > 0.0 else -1.0, 0.0])
            texcoords.extend([t, 0.0 if radial == inner_r else 1.0])
            colors.extend([220, 220, 220, 255])

    seg: int
    for seg in range(segments):
        base: int = seg * 4
        nxt: int = base + 4
        # bottom / top wound for outward normals (-Y / +Y)
        indices.extend([base + 0, nxt + 1, base + 1, base + 0, nxt + 0, nxt + 1])
        indices.extend([base + 2, nxt + 3, nxt + 2, base + 2, base + 3, nxt + 3])
        indices.extend([base + 0, nxt + 0, nxt + 2, base + 0, nxt + 2, base + 2])
        indices.extend([base + 1, base + 3, nxt + 3, base + 1, nxt + 3, nxt + 1])

    last: int = segments * 4
    indices.extend([0, 2, 3, 0, 3, 1])
    indices.extend([last + 0, last + 1, last + 3, last + 0, last + 3, last + 2])

    return CornerProxyGeometry(
        positions=tuple(positions),
        normals=tuple(normals),
        texcoords=tuple(texcoords),
        colors=tuple(colors),
        indices=tuple(indices),
        segments=segments,
    )


def make_corner_proxy_mesh(
    corner: CornerPiece, width_m: float
) -> tuple[Mesh, tuple[Any, ...]]:
    """upload a corner arc ribbon; requires an active gl context"""
    geo: CornerProxyGeometry = build_corner_proxy_geometry(corner, width_m)
    vertices_buf: Any = ffi.new("float[]", list(geo.positions))
    normals_buf: Any = ffi.new("float[]", list(geo.normals))
    texcoords_buf: Any = ffi.new("float[]", list(geo.texcoords))
    colors_buf: Any = ffi.new("unsigned char[]", list(geo.colors))
    indices_buf: Any = ffi.new("unsigned short[]", list(geo.indices))
    mesh_cdata: Any = ffi.new("Mesh *")
    mesh_cdata.vertexCount = geo.vertex_count
    mesh_cdata.triangleCount = geo.triangle_count
    mesh_cdata.vertices = vertices_buf
    mesh_cdata.normals = normals_buf
    mesh_cdata.texcoords = texcoords_buf
    mesh_cdata.colors = colors_buf
    mesh_cdata.indices = indices_buf
    mesh: Mesh = mesh_cdata[0]
    upload_mesh(mesh, False)
    keep_alive: tuple[Any, ...] = (
        mesh_cdata,
        vertices_buf,
        normals_buf,
        texcoords_buf,
        colors_buf,
        indices_buf,
    )
    return mesh, keep_alive


def ensure_road_proxy(
    catalog: RoadPartsCatalog,
    stem: str,
    *,
    width_m: float,
    has_mesh: Callable[[MeshId], bool],
    register_mesh: Callable[[MeshId, PreparedMesh], None],
    shader: Shader,
) -> MeshId:
    """register a proxy mesh for stem if missing; return its mesh id"""
    mesh_id: MeshId = road_proxy_mesh_id(stem, width_m)
    if has_mesh(mesh_id):
        return mesh_id
    corner: CornerPiece | None = catalog.corner_for_stem(stem)
    prepared: PreparedMesh
    if corner is not None:
        mesh: Mesh
        keep_alive: tuple[Any, ...]
        mesh, keep_alive = make_corner_proxy_mesh(corner, width_m)
        base: PreparedMesh = prepare_mesh(mesh, shader)
        prepared = PreparedMesh(
            mesh=base.mesh,
            material=base.material,
            local_bounds=base.local_bounds,
            keep_alive=keep_alive,
        )
    else:
        length_m: float = catalog.length_for_stem(stem)
        prepared = prepare_mesh(make_road_proxy_mesh(length_m, width_m), shader)
    register_mesh(mesh_id, prepared)
    return mesh_id


def placement_transform(
    x: float,
    z: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
    *,
    y: float = 0.0,
    pitch_deg: float = 0.0,
) -> Transform:
    """usage-local transform for a road piece (y=0 is the work plane)"""
    return Transform(
        Vector3(x, y, z),
        quaternion_from_yaw_pitch_roll(
            radians(yaw_deg), radians(pitch_deg), radians(roll_deg)
        ),
        Vector3(1.0, 1.0, 1.0),
    )
