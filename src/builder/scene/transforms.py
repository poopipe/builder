"""local and world transform matrices for scene nodes"""

from __future__ import annotations

from pyray import (
    Matrix,
    Transform,
    Vector3,
    matrix_invert,
    matrix_multiply,
    matrix_scale,
    matrix_translate,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
    vector3_length,
)

from builder.scene.scene_types import Node, Quaternion


def transform_matrix(transform: Transform) -> Matrix:
    """build model matrix from raylib transform (scale then rotate then translate)

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
    """return world matrix for node by walking parent local transforms"""
    node: Node = nodes[node_id]
    local: Matrix = transform_matrix(node.transform)
    if node.parent_id is None:
        return local
    # multiply(local, parent) => math parent@local (see transform_matrix note)
    return matrix_multiply(local, world_matrix(nodes, node.parent_id))


def matrix_translation(matrix: Matrix) -> Vector3:
    """return translation component of a raylib matrix"""
    return Vector3(matrix.m12, matrix.m13, matrix.m14)


def transform_direction(matrix: Matrix, direction: Vector3) -> Vector3:
    """apply the linear part of a matrix to a direction (ignores translation)"""
    return Vector3(
        matrix.m0 * direction.x + matrix.m4 * direction.y + matrix.m8 * direction.z,
        matrix.m1 * direction.x + matrix.m5 * direction.y + matrix.m9 * direction.z,
        matrix.m2 * direction.x + matrix.m6 * direction.y + matrix.m10 * direction.z,
    )


def world_delta_to_local(
    nodes: dict[str, Node],
    node_id: str,
    delta_world: Vector3,
) -> Vector3:
    """express a world-space translation delta in the node's parent-local axes

    root nodes store translation in world space already, so the delta is unchanged
    """
    node: Node = nodes[node_id]
    if node.parent_id is None:
        return delta_world
    return transform_direction(
        matrix_invert(world_matrix(nodes, node.parent_id)),
        delta_world,
    )


def matrix_scale_lengths(matrix: Matrix) -> Vector3:
    """return per-axis scale from the lengths of a matrix's basis columns"""
    return Vector3(
        vector3_length(Vector3(matrix.m0, matrix.m1, matrix.m2)),
        vector3_length(Vector3(matrix.m4, matrix.m5, matrix.m6)),
        vector3_length(Vector3(matrix.m8, matrix.m9, matrix.m10)),
    )


def world_rotation(nodes: dict[str, Node], node_id: str) -> Quaternion:
    """return world-space rotation by composing local rotations up the chain"""
    node: Node = nodes[node_id]
    if node.parent_id is None:
        return node.transform.rotation
    # world = parent_world @ local, so rotations compose parent-first
    return quaternion_multiply(
        world_rotation(nodes, node.parent_id), node.transform.rotation
    )


def local_transform_under_parent(
    nodes: dict[str, Node],
    child_id: str,
    new_parent_id: str | None,
) -> Transform:
    """return the local transform that keeps a node's world pose under a parent

    position and scale come from the exact world matrices; rotation is recovered
    by dividing the child's world rotation by the parent's, which is exact for
    uniform scale and a stable approximation otherwise
    """
    child_world: Matrix = world_matrix(nodes, child_id)
    child_world_rotation: Quaternion = world_rotation(nodes, child_id)
    if new_parent_id is None:
        return Transform(
            matrix_translation(child_world),
            child_world_rotation,
            matrix_scale_lengths(child_world),
        )
    parent_world: Matrix = world_matrix(nodes, new_parent_id)
    # multiply(child_world, inverse(parent_world)) => math inv(parent)@child
    local: Matrix = matrix_multiply(child_world, matrix_invert(parent_world))
    parent_world_rotation: Quaternion = world_rotation(nodes, new_parent_id)
    local_rotation: Quaternion = quaternion_multiply(
        quaternion_invert(parent_world_rotation), child_world_rotation
    )
    return Transform(
        matrix_translation(local),
        local_rotation,
        matrix_scale_lengths(local),
    )
