"""Instanced draw helpers for scene nodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pyray import draw_mesh_instanced, ffi

from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.scene.transforms import world_matrix
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


def group_nodes_by_mesh(nodes: Sequence[Node]) -> dict[MeshId, list[Node]]:
    """ bucket drawable nodes by mesh id for one draw call per mesh """
    groups: dict[MeshId, list[Node]] = defaultdict(list)
    node: Node
    for node in nodes:
        if node.mesh_id is None:
            continue
        groups[node.mesh_id].append(node)
    return groups


def draw_nodes_instanced(table: MeshTable, scene: Scene) -> None:
    """ draw all meshed nodes with draw_mesh_instanced, one batch per mesh """
    mesh_id: MeshId
    group: list[Node]
    for mesh_id, group in group_nodes_by_mesh(scene.all_nodes()).items():
        if not group:
            continue
        prepared: PreparedMesh = table.get(mesh_id)
        count: int = len(group)
        transforms = ffi.new("Matrix[]", count)
        index: int
        node: Node
        for index, node in enumerate(group):
            transforms[index] = world_matrix(scene.nodes, node.id)
        draw_mesh_instanced(
            prepared.mesh,
            prepared.material,
            ffi.cast("Matrix *", transforms),
            count,
        )
