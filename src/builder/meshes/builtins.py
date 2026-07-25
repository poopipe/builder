"""builtin mesh geometry (CPU / raylib gen only — not prepared for draw)"""

from __future__ import annotations

from pyray import (
    BoundingBox,
    Mesh,
    gen_mesh_cone,
    gen_mesh_cube,
    gen_mesh_cylinder,
    gen_mesh_sphere,
    gen_mesh_torus,
    get_mesh_bounding_box,
    update_mesh_buffer,
)


def center_mesh(mesh: Mesh) -> Mesh:
    """shift vertices so the bounding-box center sits at the origin

    raylib cylinder/cone sit on Y=0; cube/sphere/torus are already centered.
    GenMesh uploads to the GPU, so the vertex VBO is refreshed after the shift
    """
    bounds: BoundingBox = get_mesh_bounding_box(mesh)
    ox: float = (float(bounds.min.x) + float(bounds.max.x)) * 0.5
    oy: float = (float(bounds.min.y) + float(bounds.max.y)) * 0.5
    oz: float = (float(bounds.min.z) + float(bounds.max.z)) * 0.5
    if abs(ox) < 1e-12 and abs(oy) < 1e-12 and abs(oz) < 1e-12:
        return mesh
    count: int = int(mesh.vertexCount)
    index: int
    for index in range(count):
        mesh.vertices[index * 3] -= ox
        mesh.vertices[index * 3 + 1] -= oy
        mesh.vertices[index * 3 + 2] -= oz
    update_mesh_buffer(mesh, 0, mesh.vertices, count * 3 * 4, 0)
    return mesh


def make_cube(size: float = 1.5) -> Mesh:
    """create a cube mesh. Caller owns unload"""
    return gen_mesh_cube(size, size, size)


def make_cylinder(
    radius: float = 0.5, height: float = 1.5, slices: int = 16
) -> Mesh:
    """create a cylinder mesh centered on the origin. Caller owns unload"""
    return center_mesh(gen_mesh_cylinder(radius, height, slices))


def make_cone(
    radius: float = 0.5, height: float = 1.5, slices: int = 16
) -> Mesh:
    """create a cone mesh centered on the origin. Caller owns unload"""
    return center_mesh(gen_mesh_cone(radius, height, slices))


def make_torus(
    radius: float = 0.35,
    size: float = 1.5,
    rad_seg: int = 16,
    sides: int = 16,
) -> Mesh:
    """create a torus mesh. Caller owns unload

    raylib treats radius as the hole/tube ratio (clamped 0.1..1) and size as
    overall extent, matching the other ~1.5 builtins
    """
    return gen_mesh_torus(radius, size, rad_seg, sides)


def make_sphere(radius: float = 0.75, rings: int = 16, slices: int = 16) -> Mesh:
    """create a sphere mesh. Caller owns unload"""
    return gen_mesh_sphere(radius, rings, slices)
