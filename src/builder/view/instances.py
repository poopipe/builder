"""Instanced draw helpers for scene nodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pyray import (
    Matrix,
    Transform,
    draw_mesh_instanced,
    ffi,
    matrix_multiply,
    matrix_scale,
    matrix_translate,
    quaternion_to_matrix,
)

from builder.scene.scene_types import MeshId, Node
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


def transform_matrix(transform: Transform) -> Matrix:
    """ build model matrix from raylib transform (scale then rotate then translate) """
    scale: Matrix = matrix_scale(
        transform.scale.x, transform.scale.y, transform.scale.z
    )
    rotation: Matrix = quaternion_to_matrix(transform.rotation)
    translation: Matrix = matrix_translate(
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    )
    return matrix_multiply(translation, matrix_multiply(rotation, scale))


def group_nodes_by_mesh(nodes: Sequence[Node]) -> dict[MeshId, list[Node]]:
    """ bucket nodes by mesh id for one draw call per mesh """
    groups: dict[MeshId, list[Node]] = defaultdict(list)
    node: Node
    for node in nodes:
        groups[node.mesh_id].append(node)
    return groups


def draw_nodes_instanced(table: MeshTable, nodes: Sequence[Node]) -> None:
    """ draw all nodes with draw_mesh_instanced, one batch per mesh """
    mesh_id: MeshId
    group: list[Node]
    for mesh_id, group in group_nodes_by_mesh(nodes).items():
        if not group:
            continue
        prepared: PreparedMesh = table.get(mesh_id)
        count: int = len(group)
        transforms = ffi.new("Matrix[]", count)
        index: int
        node: Node
        for index, node in enumerate(group):
            transforms[index] = transform_matrix(node.transform)
        draw_mesh_instanced(
            prepared.mesh,
            prepared.material,
            ffi.cast("Matrix *", transforms),
            count,
        )
