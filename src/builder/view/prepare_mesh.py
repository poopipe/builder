"""Prepare meshes for GPU drawing (material + shader binding)."""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Material, Mesh, Shader, load_material_default


@dataclass(frozen=True)
class PreparedMesh:
    """A mesh ready for instanced draw with the bound material/shader."""

    mesh: Mesh
    material: Material


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
    return PreparedMesh(mesh=mesh, material=material)
