"""pure transform operations for group editing"""

from __future__ import annotations

from pyray import (
    Transform,
    Vector3,
    quaternion_multiply,
    vector3_add,
    vector3_rotate_by_quaternion,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import Quaternion


def translate_transform(transform: Transform, delta: Vector3) -> Transform:
    """return transform with delta added to its translation

    delta must already be in the same space as transform.translation
    """
    return Transform(
        vector3_add(transform.translation, delta),
        transform.rotation,
        transform.scale,
    )


def translate_transform_local(
    transform: Transform,
    offset_local: Vector3,
) -> Transform:
    """return transform moved along its own axes"""
    return Transform(
        vector3_add(
            transform.translation,
            vector3_rotate_by_quaternion(offset_local, transform.rotation),
        ),
        transform.rotation,
        transform.scale,
    )


def set_transform_translation(
    transform: Transform,
    translation: Vector3,
) -> Transform:
    """return transform with its translation replaced"""
    return Transform(translation, transform.rotation, transform.scale)


def set_transform_rotation(
    transform: Transform,
    rotation: Quaternion,
) -> Transform:
    """return transform with its rotation replaced"""
    return Transform(transform.translation, rotation, transform.scale)


def set_transform_scale(transform: Transform, scale: Vector3) -> Transform:
    """return transform with its scale replaced"""
    return Transform(transform.translation, transform.rotation, scale)


def rotate_transform_world(
    transform: Transform,
    rotation: Quaternion,
    pivot: Vector3,
) -> Transform:
    """return transform rotated in world space about pivot"""
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
    """return transform with local-space rotation about its origin"""
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
    """return transform with uniform scale about pivot"""
    offset: Vector3 = vector3_subtract(transform.translation, pivot)
    scaled_offset: Vector3 = vector3_scale(offset, factor)
    scale: Vector3 = vector3_scale(transform.scale, factor)
    return Transform(
        vector3_add(pivot, scaled_offset),
        transform.rotation,
        scale,
    )
