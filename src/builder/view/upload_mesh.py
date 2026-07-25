"""Upload imported CPU meshes into raylib GPU meshes."""

from __future__ import annotations

from typing import Any

from pyray import Mesh, Shader, ffi, upload_mesh

from builder.io.mesh_import import ImportedMesh
from builder.view.prepare_mesh import PreparedMesh, prepare_mesh


def upload_imported_mesh(imported: ImportedMesh, shader: Shader) -> PreparedMesh:
    """ build a GPU mesh from an imported triangle soup and bind the lighting shader """
    vertex_count: int = imported.vertex_count
    if vertex_count < 3 or vertex_count % 3 != 0:
        raise ValueError(
            f"imported mesh needs a multiple of 3 vertices, got {vertex_count}"
        )
    vertices: Any = ffi.new("float[]", list(imported.positions))
    normals: Any = ffi.new("float[]", list(imported.normals))
    texcoords: Any = ffi.new("float[]", list(imported.texcoords))
    # zero-init so optional attribute pointers stay NULL across raylib versions
    mesh_cdata: Any = ffi.new("Mesh *")
    mesh_cdata.vertexCount = vertex_count
    mesh_cdata.triangleCount = imported.triangle_count
    mesh_cdata.vertices = vertices
    mesh_cdata.normals = normals
    mesh_cdata.texcoords = texcoords
    mesh: Mesh = mesh_cdata[0]
    upload_mesh(mesh, False)
    prepared: PreparedMesh = prepare_mesh(mesh, shader)
    # keep cpu buffers + Mesh cdata alive for the lifetime of the prepared mesh
    return PreparedMesh(
        mesh=prepared.mesh,
        material=prepared.material,
        local_bounds=prepared.local_bounds,
        keep_alive=(mesh_cdata, vertices, normals, texcoords),
    )
