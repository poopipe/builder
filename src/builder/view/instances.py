"""Instanced draw helpers for scene nodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pyray import (
    Matrix,
    Vector3,
    draw_mesh_instanced,
    ffi,
    matrix_multiply,
    matrix_rotate_xyz,
    matrix_scale,
    matrix_translate,
)

from builder.scene.scene_types import MeshId, Node, Transform
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


def transform_matrix(transform: Transform) -> Matrix:
    """Build a model matrix from TRS (scale → rotate → translate)."""
    scale: Matrix = matrix_scale(
        transform.scale.x, transform.scale.y, transform.scale.z
    )
    rotation: Matrix = matrix_rotate_xyz(
        Vector3(transform.rotation.x, transform.rotation.y, transform.rotation.z)
    )
    translation: Matrix = matrix_translate(
        transform.position.x, transform.position.y, transform.position.z
    )
    return matrix_multiply(translation, matrix_multiply(rotation, scale))


def group_nodes_by_mesh(nodes: Sequence[Node]) -> dict[MeshId, list[Node]]:
    """Bucket nodes by mesh id for one draw call per mesh."""
    groups: dict[MeshId, list[Node]] = defaultdict(list)
    node: Node
    for node in nodes:
        groups[node.mesh_id].append(node)
    return groups


def draw_nodes_instanced(table: MeshTable, nodes: Sequence[Node]) -> None:
    """Draw all nodes with ``draw_mesh_instanced``, one batch per mesh."""
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
