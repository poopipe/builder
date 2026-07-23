"""Local and world transform matrices for scene nodes."""

from __future__ import annotations

from pyray import (
    Matrix,
    Transform,
    Vector3,
    matrix_multiply,
    matrix_scale,
    matrix_translate,
    quaternion_to_matrix,
)

from builder.scene.scene_types import Node


def transform_matrix(transform: Transform) -> Matrix:
    """ build model matrix from raylib transform (scale then rotate then translate)

    pyray's matrix_multiply(a, b) composes as math b@a, so arguments are ordered
    scale, rotation, translation to yield column-vector T@R@S
    """
    scale: Matrix = matrix_scale(
        transform.scale.x, transform.scale.y, transform.scale.z
    )
    rotation: Matrix = quaternion_to_matrix(transform.rotation)
    translation: Matrix = matrix_translate(
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    )
    return matrix_multiply(matrix_multiply(scale, rotation), translation)


def world_matrix(nodes: dict[str, Node], node_id: str) -> Matrix:
    """ return world matrix for node by walking parent local transforms """
    node: Node = nodes[node_id]
    local: Matrix = transform_matrix(node.transform)
    if node.parent_id is None:
        return local
    # multiply(local, parent) => math parent@local (see transform_matrix note)
    return matrix_multiply(local, world_matrix(nodes, node.parent_id))


def matrix_translation(matrix: Matrix) -> Vector3:
    """ return translation component of a raylib matrix """
    return Vector3(matrix.m12, matrix.m13, matrix.m14)
