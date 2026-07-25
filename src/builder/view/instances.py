"""Instanced draw helpers for scene nodes."""

from __future__ import annotations

from pyray import Matrix, draw_mesh_instanced, ffi, matrix_identity

from builder.scene.scene import Scene
from builder.view.draw_cache import DrawCache, MeshInstanceBatch, sync_draw_cache
from builder.view.lighting import Lighting
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


def draw_nodes_instanced(
    table: MeshTable,
    scene: Scene,
    cache: DrawCache,
    lighting: Lighting,
) -> None:
    """ draw meshed nodes: one call per (mesh, parent), parent world as a uniform """
    sync_draw_cache(scene, cache)
    identity: Matrix = matrix_identity()
    batch: MeshInstanceBatch
    for batch in cache.batches.values():
        if batch.count == 0 or not table.has_mesh(batch.mesh_id):
            continue
        prepared: PreparedMesh = table.get(batch.mesh_id)
        parent: Matrix = identity
        if batch.parent_id is not None:
            parent = cache.world[batch.parent_id]
        lighting.set_parent_transform(parent)
        draw_mesh_instanced(
            prepared.mesh,
            prepared.material,
            ffi.cast("Matrix *", batch.transforms),
            batch.count,
        )
