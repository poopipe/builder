"""Axis-aligned bounds helpers for picking."""

from __future__ import annotations

from pyray import BoundingBox, Matrix, Vector3, vector3_transform


def bounding_box_corners(bounds: BoundingBox) -> list[Vector3]:
    """ return the eight corners of a local bounding box """
    minimum: Vector3 = bounds.min
    maximum: Vector3 = bounds.max
    return [
        Vector3(minimum.x, minimum.y, minimum.z),
        Vector3(maximum.x, minimum.y, minimum.z),
        Vector3(minimum.x, maximum.y, minimum.z),
        Vector3(maximum.x, maximum.y, minimum.z),
        Vector3(minimum.x, minimum.y, maximum.z),
        Vector3(maximum.x, minimum.y, maximum.z),
        Vector3(minimum.x, maximum.y, maximum.z),
        Vector3(maximum.x, maximum.y, maximum.z),
    ]


def transform_bounding_box(bounds: BoundingBox, matrix: Matrix) -> BoundingBox:
    """ return a world aabb that contains the transformed local bounds """
    corners: list[Vector3] = bounding_box_corners(bounds)
    first: Vector3 = vector3_transform(corners[0], matrix)
    min_x: float = first.x
    min_y: float = first.y
    min_z: float = first.z
    max_x: float = first.x
    max_y: float = first.y
    max_z: float = first.z
    corner: Vector3
    for corner in corners[1:]:
        point: Vector3 = vector3_transform(corner, matrix)
        min_x = min(min_x, point.x)
        min_y = min(min_y, point.y)
        min_z = min(min_z, point.z)
        max_x = max(max_x, point.x)
        max_y = max(max_y, point.y)
        max_z = max(max_z, point.z)
    return BoundingBox(
        Vector3(min_x, min_y, min_z),
        Vector3(max_x, max_y, max_z),
    )
