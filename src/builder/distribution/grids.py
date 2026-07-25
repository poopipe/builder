"""grid distribution functions — each returns a list of transforms"""

from __future__ import annotations

from math import acos, cos, pi, radians, sin

from pyray import (
    Transform,
    Vector3,
    Vector4,
    quaternion_from_axis_angle,
    quaternion_identity,
    vector3_add,
    vector3_cross_product,
    vector3_dot_product,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import transform_at

type AxisIndex = int


def clamp_axis(axis: AxisIndex) -> AxisIndex:
    """clamp an axis index to 0..2 (x/y/z)"""
    return max(0, min(2, int(axis)))


def axis_vector(axis: AxisIndex, value: float) -> Vector3:
    """return a vector with value along the given axis"""
    if axis == 0:
        return Vector3(value, 0.0, 0.0)
    if axis == 1:
        return Vector3(0.0, value, 0.0)
    return Vector3(0.0, 0.0, value)


def axis_component(position: Vector3, axis: AxisIndex) -> float:
    """return the position component on the given axis"""
    if axis == 0:
        return float(position.x)
    if axis == 1:
        return float(position.y)
    return float(position.z)


def ring_direction(axis: AxisIndex, angle_rad: float) -> Vector3:
    """unit direction in the plane perpendicular to the cylinder axis"""
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)
    if axis == 0:
        return Vector3(0.0, c, s)
    if axis == 1:
        return Vector3(c, 0.0, s)
    return Vector3(c, s, 0.0)


def rotation_from_z_to(direction: Vector3) -> Vector4:
    """quaternion rotating +Z onto direction"""
    target: Vector3 = vector3_normalize(direction)
    z: Vector3 = Vector3(0.0, 0.0, 1.0)
    dot: float = vector3_dot_product(z, target)
    if dot > 0.999999:
        return quaternion_identity()
    if dot < -0.999999:
        return quaternion_from_axis_angle(Vector3(1.0, 0.0, 0.0), pi)
    axis: Vector3 = vector3_normalize(vector3_cross_product(z, target))
    angle: float = acos(max(-1.0, min(1.0, dot)))
    return quaternion_from_axis_angle(axis, angle)


def align_build_from(
    transforms: list[Transform],
    origin: Vector3,
    build_from: AxisIndex,
) -> list[Transform]:
    """shift so the build-from axis min of positions equals origin on that axis"""
    if not transforms:
        return transforms
    axis: AxisIndex = clamp_axis(build_from)
    min_val: float = min(
        axis_component(transform.translation, axis) for transform in transforms
    )
    delta: float = axis_component(origin, axis) - min_val
    if abs(delta) < 1e-12:
        return transforms
    offset: Vector3 = axis_vector(axis, delta)
    aligned: list[Transform] = []
    transform: Transform
    for transform in transforms:
        aligned.append(
            Transform(
                vector3_add(transform.translation, offset),
                transform.rotation,
                transform.scale,
            )
        )
    return aligned


def grid_transforms(
    origin: Vector3,
    count_x: int = 3,
    count_y: int = 1,
    count_z: int = 3,
    spacing_x: float = 2.0,
    spacing_y: float = 2.0,
    spacing_z: float = 2.0,
    build_from: AxisIndex = 1,
) -> list[Transform]:
    """return transforms for a regular XYZ lattice

    centered on the non-build axes; flush to origin on build_from (default Y)
    """
    step_x: Vector3 = Vector3(spacing_x, 0.0, 0.0)
    step_y: Vector3 = Vector3(0.0, spacing_y, 0.0)
    step_z: Vector3 = Vector3(0.0, 0.0, spacing_z)
    start: Vector3 = vector3_subtract(
        origin,
        vector3_add(
            vector3_add(
                vector3_scale(step_x, (count_x - 1) * 0.5),
                vector3_scale(step_y, (count_y - 1) * 0.5),
            ),
            vector3_scale(step_z, (count_z - 1) * 0.5),
        ),
    )
    transforms: list[Transform] = []
    ix: int
    iy: int
    iz: int
    for iz in range(count_z):
        for iy in range(count_y):
            for ix in range(count_x):
                position: Vector3 = vector3_add(
                    start,
                    vector3_add(
                        vector3_add(
                            vector3_scale(step_x, float(ix)),
                            vector3_scale(step_y, float(iy)),
                        ),
                        vector3_scale(step_z, float(iz)),
                    ),
                )
                transforms.append(transform_at(position))
    return align_build_from(transforms, origin, build_from)


def radial_grid_transforms(
    origin: Vector3,
    radius: float,
    spacing: float,
    face_center: bool = False,
    axis: AxisIndex = 1,
    count_height: int = 1,
    spacing_height: float = 2.0,
    count_radius: int = 1,
    spacing_radius: float = 2.0,
    build_from: AxisIndex = 1,
) -> list[Transform]:
    """return transforms for stacked concentric rings around a cylinder axis

    spacing is the angular step in degrees.
    count_radius adds concentric rings starting at radius, spaced by spacing_radius.
    count_height stacks copies along axis.
    when face_center is set each transform's +Z points toward the cylinder axis.
    """
    cylinder: AxisIndex = clamp_axis(axis)
    height_count: int = max(1, int(count_height))
    radius_count: int = max(1, int(count_radius))
    transforms: list[Transform] = []
    ih: int
    ir: int
    for ih in range(height_count):
        height: float = (float(ih) - (height_count - 1) * 0.5) * spacing_height
        height_offset: Vector3 = axis_vector(cylinder, height)
        for ir in range(radius_count):
            ring_radius: float = radius + float(ir) * spacing_radius
            angle_deg: float = 0.0
            while angle_deg < 360.0:
                angle_rad: float = radians(angle_deg)
                direction: Vector3 = ring_direction(cylinder, angle_rad)
                position: Vector3 = vector3_add(
                    origin,
                    vector3_add(
                        vector3_scale(direction, ring_radius), height_offset
                    ),
                )
                if face_center:
                    # +Z toward the cylinder axis = opposite the ring direction
                    inward: Vector3 = vector3_scale(direction, -1.0)
                    transforms.append(
                        Transform(
                            position,
                            rotation_from_z_to(inward),
                            Vector3(1.0, 1.0, 1.0),
                        )
                    )
                else:
                    transforms.append(transform_at(position))
                angle_deg += spacing
    return align_build_from(transforms, origin, build_from)
