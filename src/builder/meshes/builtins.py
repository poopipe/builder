"""Builtin mesh geometry (CPU / raylib gen only — not prepared for draw)."""

from __future__ import annotations

from pyray import Mesh, gen_mesh_cube, gen_mesh_cylinder


def make_cube(size: float = 1.5) -> Mesh:
    """Create a cube mesh. Caller owns unload."""
    return gen_mesh_cube(size, size, size)


def make_cylinder(
    radius: float = 0.5, height: float = 1.5, slices: int = 16
) -> Mesh:
    """Create a cylinder mesh. Caller owns unload."""
    return gen_mesh_cylinder(radius, height, slices)
