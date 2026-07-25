"""grid distribution functions — each returns a list of transforms"""

from __future__ import annotations

from math import atan2, cos, radians, sin

from pyray import (
    Transform,
    Vector3,
    quaternion_from_axis_angle,
    vector3_add,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import transform_at


def horizontal_grid_transforms(
    origin: Vector3,
    count_x: int = 3,
    count_z: int = 3,
    spacing: float = 2.0,
) -> list[Transform]:
    """return transforms for grid in XZ centered at origin"""
    step_x: Vector3 = Vector3(spacing, 0.0, 0.0)
    step_z: Vector3 = Vector3(0.0, 0.0, spacing)
    start: Vector3 = vector3_subtract(
        origin,
        vector3_add(
            vector3_scale(step_x, (count_x - 1) * 0.5),
            vector3_scale(step_z, (count_z - 1) * 0.5),
        ),
    )
    transforms: list[Transform] = []
    ix: int
    iz: int
    for iz in range(count_z):
        for ix in range(count_x):
            position: Vector3 = vector3_add(
                start,
                vector3_add(
                    vector3_scale(step_x, float(ix)),
                    vector3_scale(step_z, float(iz)),
                ),
            )
            transforms.append(transform_at(position))
    return transforms


def radial_grid_transforms(
    origin: Vector3,
    radius: float,
    spacing: float,
    face_center: bool = False,
) -> list[Transform]:
    """return transforms for radial grid in XZ centered at origin

    when face_center is set each transform is yawed so its +Z axis points at origin
    """
    transforms: list[Transform] = []
    angle_deg: float = 0.0
    while angle_deg < 360.0:
        angle_rad: float = radians(angle_deg)
        direction: Vector3 = Vector3(cos(angle_rad), 0.0, sin(angle_rad))
        position: Vector3 = vector3_add(origin, vector3_scale(direction, radius))
        if face_center:
            # +Z should point inward (toward origin), i.e. along -direction
            yaw: float = atan2(-direction.x, -direction.z)
            transforms.append(
                Transform(
                    position,
                    quaternion_from_axis_angle(Vector3(0.0, 1.0, 0.0), yaw),
                    Vector3(1.0, 1.0, 1.0),
                )
            )
        else:
            transforms.append(transform_at(position))
        angle_deg += spacing
    return transforms
