"""Pure transform operations for group editing."""

from __future__ import annotations

from dataclasses import replace

from pyray import (
    Transform,
    Vector3,
    quaternion_multiply,
    vector3_add,
    vector3_rotate_by_quaternion,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import Node, Quaternion


def with_transform(node: Node, transform: Transform) -> Node:
    """ return a copy of node with a new transform """
    return replace(node, transform=transform)


def translate_transform(transform: Transform, delta_world: Vector3) -> Transform:
    """ return transform with world-space translation applied """
    return Transform(
        vector3_add(transform.translation, delta_world),
        transform.rotation,
        transform.scale,
    )


def rotate_transform_world(
    transform: Transform,
    rotation: Quaternion,
    pivot: Vector3,
) -> Transform:
    """ return transform rotated in world space about pivot """
    offset: Vector3 = vector3_subtract(transform.translation, pivot)
    rotated_offset: Vector3 = vector3_rotate_by_quaternion(offset, rotation)
    return Transform(
        vector3_add(pivot, rotated_offset),
        quaternion_multiply(rotation, transform.rotation),
        transform.scale,
    )


def rotate_transform_local(
    transform: Transform,
    rotation: Quaternion,
) -> Transform:
    """ return transform with local-space rotation about its origin """
    return Transform(
        transform.translation,
        quaternion_multiply(transform.rotation, rotation),
        transform.scale,
    )


def scale_transform_uniform(
    transform: Transform,
    factor: float,
    pivot: Vector3,
) -> Transform:
    """ return transform with uniform scale about pivot """
    offset: Vector3 = vector3_subtract(transform.translation, pivot)
    scaled_offset: Vector3 = vector3_scale(offset, factor)
    scale: Vector3 = vector3_scale(transform.scale, factor)
    return Transform(
        vector3_add(pivot, scaled_offset),
        transform.rotation,
        scale,
    )
