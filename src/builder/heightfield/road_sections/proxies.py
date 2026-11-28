"""on-demand proxy meshes for road piece stems"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, hypot, radians, sin
from typing import Any

from pyray import Mesh, Shader, Transform, Vector3, ffi, gen_mesh_cube, upload_mesh

from builder.heightfield.road_sections.kit_widths import PieceSnapPoints
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


def _rotate_cw_xz(vx: float, vz: float, deg: float) -> tuple[float, float]:
    r: float = radians(deg)
    c: float = cos(r)
    s: float = sin(r)
    return vx * c + vz * s, -vx * s + vz * c


def _ring_indices(segments: int) -> list[int]:
    """side quads + end caps for a 4-vert-per-ring ribbon"""
    indices: list[int] = []
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
    return indices


def _append_ring_verts(
    positions: list[float],
    normals: list[float],
    texcoords: list[float],
    colors: list[int],
    *,
    left_x: float,
    left_z: float,
    right_x: float,
    right_z: float,
    half_h: float,
    t: float,
) -> None:
    """four verts: left/right × bottom/top"""
    for x, z, y, v in (
        (left_x, left_z, -half_h, 0.0),
        (right_x, right_z, -half_h, 1.0),
        (left_x, left_z, half_h, 0.0),
        (right_x, right_z, half_h, 1.0),
    ):
        positions.extend([x, y, z])
        normals.extend([0.0, 1.0 if y > 0.0 else -1.0, 0.0])
        texcoords.extend([t, v])
        colors.extend([220, 220, 220, 255])


def build_corner_proxy_geometry(
    corner: CornerPiece,
    width_m: float,
    *,
    snaps: PieceSnapPoints | None = None,
) -> CornerProxyGeometry:
    """cw arc ribbon in p3d local space

    with snaps: begin/end rings pin to catalogue lb/pb and le/pe; origin is
    the catalog mesh origin. without snaps: ideal mid-arc ribbon.
    """
    if snaps is not None:
        return build_corner_proxy_from_snaps(corner, snaps)
    return build_ideal_corner_proxy_geometry(corner, width_m)


def build_ideal_corner_proxy_geometry(
    corner: CornerPiece,
    width_m: float,
) -> CornerProxyGeometry:
    """cw arc ribbon; entry heading +Z; origin at mid-arc"""
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

    ring: int
    for ring in range(segments + 1):
        t: float = float(ring) / float(segments)
        theta: float = radians(angle_deg * t)
        c: float = cos(theta)
        s: float = sin(theta)
        left_x: float = radius - inner_r * c - mid_x
        left_z: float = inner_r * s - mid_z
        right_x: float = radius - outer_r * c - mid_x
        right_z: float = outer_r * s - mid_z
        _append_ring_verts(
            positions,
            normals,
            texcoords,
            colors,
            left_x=left_x,
            left_z=left_z,
            right_x=right_x,
            right_z=right_z,
            half_h=half_h,
            t=t,
        )

    return CornerProxyGeometry(
        positions=tuple(positions),
        normals=tuple(normals),
        texcoords=tuple(texcoords),
        colors=tuple(colors),
        indices=tuple(_ring_indices(segments)),
        segments=segments,
    )


def build_corner_proxy_from_snaps(
    corner: CornerPiece,
    snaps: PieceSnapPoints,
) -> CornerProxyGeometry:
    """cw ribbon pinned to catalogue snaps; origin is p3d origin"""
    angle_deg: float = max(float(corner.angle_deg), 1.0)
    radius: float = max(float(corner.radius_m), 0.5)
    half_h: float = road_proxy_height_m * 0.5
    segments: int = corner_segment_count(angle_deg)
    lb: tuple[float, float, float] = snaps.lb
    le: tuple[float, float, float] = snaps.le
    pb: tuple[float, float, float] = snaps.pb
    pe: tuple[float, float, float] = snaps.pe
    bx: float
    bz: float
    bx, bz = snaps.begin_mid_xz()
    # entry heading from begin edge (left→right × up → forward)
    right_x: float = pb[0] - lb[0]
    right_z: float = pb[2] - lb[2]
    right_len: float = hypot(right_x, right_z)
    if right_len < 1e-6:
        right_x, right_z = 1.0, 0.0
        right_len = 1.0
    right_x /= right_len
    right_z /= right_len
    entry_x: float = -right_z
    entry_z: float = right_x
    # cw arc center from begin mid
    cx: float = bx + radius * entry_z
    cz: float = bz - radius * entry_x

    positions: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    colors: list[int] = []

    # ribbon left/right = inner/outer (same as ideal cw mesh). road-tool
    # lb/le are outer on a cw bend, pb/pe are inner — swap for winding.
    inner_b_x: float
    inner_b_z: float
    outer_b_x: float
    outer_b_z: float
    inner_e_x: float
    inner_e_z: float
    outer_e_x: float
    outer_e_z: float
    inner_b_x, inner_b_z = pb[0], pb[2]
    outer_b_x, outer_b_z = lb[0], lb[2]
    inner_e_x, inner_e_z = pe[0], pe[2]
    outer_e_x, outer_e_z = le[0], le[2]

    ring: int
    for ring in range(segments + 1):
        t: float = float(ring) / float(segments)
        left_x: float
        left_z: float
        right_px: float
        right_pz: float
        if ring == 0:
            left_x, left_z = inner_b_x, inner_b_z
            right_px, right_pz = outer_b_x, outer_b_z
        elif ring == segments:
            left_x, left_z = inner_e_x, inner_e_z
            right_px, right_pz = outer_e_x, outer_e_z
        else:
            sweep: float = angle_deg * t
            left_x, left_z = _rotate_cw_xz(
                inner_b_x - cx, inner_b_z - cz, sweep
            )
            left_x += cx
            left_z += cz
            right_px, right_pz = _rotate_cw_xz(
                outer_b_x - cx, outer_b_z - cz, sweep
            )
            right_px += cx
            right_pz += cz
        _append_ring_verts(
            positions,
            normals,
            texcoords,
            colors,
            left_x=left_x,
            left_z=left_z,
            right_x=right_px,
            right_z=right_pz,
            half_h=half_h,
            t=t,
        )

    return CornerProxyGeometry(
        positions=tuple(positions),
        normals=tuple(normals),
        texcoords=tuple(texcoords),
        colors=tuple(colors),
        indices=tuple(_ring_indices(segments)),
        segments=segments,
    )


def make_corner_proxy_mesh(
    corner: CornerPiece,
    width_m: float,
    *,
    snaps: PieceSnapPoints | None = None,
) -> tuple[Mesh, tuple[Any, ...]]:
    """upload a corner arc ribbon; requires an active gl context"""
    geo: CornerProxyGeometry = build_corner_proxy_geometry(
        corner, width_m, snaps=snaps
    )
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
        snaps: PieceSnapPoints | None = catalog.snaps_for_stem(stem)
        mesh: Mesh
        keep_alive: tuple[Any, ...]
        mesh, keep_alive = make_corner_proxy_mesh(
            corner, width_m, snaps=snaps
        )
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
