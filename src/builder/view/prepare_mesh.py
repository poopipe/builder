"""Prepare meshes for GPU drawing (material + shader binding)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyray import (
    BoundingBox,
    Material,
    Mesh,
    Shader,
    get_mesh_bounding_box,
    load_material_default,
)


@dataclass(frozen=True)
class PreparedMesh:
    """A mesh ready for instanced draw with the bound material/shader."""

    mesh: Mesh
    material: Material
    local_bounds: BoundingBox
    # ffi buffers that must outlive the Mesh (imported uploads only)
    keep_alive: tuple[Any, ...] = ()


def prepare_mesh(mesh: Mesh, shader: Shader) -> PreparedMesh:
    """Bind ``mesh`` to a default material using ``shader``.

    Does not take ownership of ``shader``. Raises if the mesh has no vertices.
    """
    vertex_count: int = int(
        getattr(mesh, "vertexCount", getattr(mesh, "vertex_count", 0))
    )
    if vertex_count <= 0:
        raise RuntimeError("mesh has no vertices")
    material: Material = load_material_default()
    material.shader = shader
    local_bounds: BoundingBox = get_mesh_bounding_box(mesh)
    return PreparedMesh(mesh=mesh, material=material, local_bounds=local_bounds)
