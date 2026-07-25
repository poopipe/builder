"""prepare meshes for GPU drawing (material + shader binding)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyray import (
    BoundingBox,
    Material,
    Mesh,
    Shader,
    ffi,
    get_mesh_bounding_box,
    load_material_default,
    unload_mesh,
)


@dataclass(frozen=True)
class PreparedMesh:
    """a mesh ready for instanced draw with the bound material/shader"""

    mesh: Mesh
    material: Material
    local_bounds: BoundingBox
    # ffi buffers that must outlive the Mesh (imported uploads only)
    keep_alive: tuple[Any, ...] = ()


def prepare_mesh(mesh: Mesh, shader: Shader) -> PreparedMesh:
    """bind ``mesh`` to a default material using ``shader``

    Does not take ownership of ``shader``. Raises if the mesh has no vertices
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


def release_prepared_mesh(prepared: PreparedMesh) -> None:
    """unload GPU mesh without freeing cffi-owned CPU attribute buffers"""
    mesh: Mesh = prepared.mesh
    if prepared.keep_alive:
        # UnloadMesh would RL_FREE these; they belong to keep_alive
        field_name: str
        for field_name in (
            "vertices",
            "texcoords",
            "texcoords2",
            "normals",
            "tangents",
            "colors",
            "indices",
            "animVertices",
            "animNormals",
            "boneIds",
            "boneWeights",
            "boneMatrices",
        ):
            try:
                setattr(mesh, field_name, ffi.NULL)
            except AttributeError:
                pass
    unload_mesh(mesh)
