"""instanced draw helpers for scene nodes"""

from __future__ import annotations

from pyray import (
    Color,
    MaterialMapIndex,
    Matrix,
    draw_mesh_instanced,
    ffi,
    matrix_identity,
)

from builder.scene.scene import Scene
from builder.ui.color_palette import palette_color
from builder.view.draw_cache import DrawCache, MeshInstanceBatch, sync_draw_cache
from builder.view.lighting import Lighting
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh

_default_mesh_albedo: Color = Color(255, 255, 255, 255)


def material_has_albedo_texture(prepared: PreparedMesh) -> bool:
    """true when an albedo texture is bound on the material"""
    return int(
        prepared.material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture.id
    ) != 0


def set_material_albedo_color(prepared: PreparedMesh, color: Color) -> None:
    """set the material albedo tint used as colDiffuse"""
    prepared.material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = color


def draw_nodes_instanced(
    table: MeshTable,
    scene: Scene,
    cache: DrawCache,
    lighting: Lighting,
) -> None:
    """draw meshed nodes: one call per (mesh, parent, color), parent world as a uniform"""
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
        tint: Color = _default_mesh_albedo
        if not material_has_albedo_texture(prepared):
            resolved: Color | None = palette_color(batch.color_index)
            if resolved is not None:
                tint = resolved
        set_material_albedo_color(prepared, tint)
        draw_mesh_instanced(
            prepared.mesh,
            prepared.material,
            ffi.cast("Matrix *", batch.transforms),
            batch.count,
        )
        set_material_albedo_color(prepared, _default_mesh_albedo)
