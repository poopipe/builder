"""yaw / pitch / roll conversions for node rotation

convention for this y-up world: yaw turns about +Y, pitch about +X, roll about +Z,
applied yaw then pitch then roll, so yaw reads as heading and roll banks the result
"""

from __future__ import annotations

from math import asin, atan2

from pyray import (
    Matrix,
    Vector3,
    quaternion_from_axis_angle,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_to_matrix,
    vector3_cross_product,
    vector3_dot_product,
    vector3_length,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import Quaternion

# beyond this the pitch is vertical and yaw/roll turn about the same world axis
pitch_gimbal_limit: float = 0.999999


def quaternion_from_yaw_pitch_roll(
    yaw: float,
    pitch: float,
    roll: float,
) -> Quaternion:
    """build a rotation from angles in radians"""
    return quaternion_multiply(
        quaternion_multiply(
            quaternion_from_axis_angle(Vector3(0.0, 1.0, 0.0), yaw),
            quaternion_from_axis_angle(Vector3(1.0, 0.0, 0.0), pitch),
        ),
        quaternion_from_axis_angle(Vector3(0.0, 0.0, 1.0), roll),
    )


def quaternion_align_y(up: Vector3, forward_hint: Vector3) -> Quaternion:
    """rotation with local +Y along up; +Z toward forward_hint on the tangent plane"""
    upward: Vector3 = vector3_normalize(up)
    projected: Vector3 = vector3_subtract(
        forward_hint,
        vector3_scale(upward, vector3_dot_product(forward_hint, upward)),
    )
    forward: Vector3
    if vector3_length(projected) < 1e-6:
        fallback: Vector3 = Vector3(0.0, 0.0, 1.0)
        if abs(vector3_dot_product(upward, fallback)) > 0.999:
            fallback = Vector3(1.0, 0.0, 0.0)
        projected = vector3_subtract(
            fallback,
            vector3_scale(upward, vector3_dot_product(fallback, upward)),
        )
    forward = vector3_normalize(projected)
    right: Vector3 = vector3_normalize(vector3_cross_product(upward, forward))
    forward = vector3_normalize(vector3_cross_product(right, upward))
    matrix: Matrix = Matrix()
    matrix.m0 = float(right.x)
    matrix.m1 = float(right.y)
    matrix.m2 = float(right.z)
    matrix.m3 = 0.0
    matrix.m4 = float(upward.x)
    matrix.m5 = float(upward.y)
    matrix.m6 = float(upward.z)
    matrix.m7 = 0.0
    matrix.m8 = float(forward.x)
    matrix.m9 = float(forward.y)
    matrix.m10 = float(forward.z)
    matrix.m11 = 0.0
    matrix.m12 = 0.0
    matrix.m13 = 0.0
    matrix.m14 = 0.0
    matrix.m15 = 1.0
    return quaternion_from_matrix(matrix)


def yaw_pitch_roll_from_quaternion(
    rotation: Quaternion,
) -> tuple[float, float, float]:
    """return (yaw, pitch, roll) radians that rebuild this rotation"""
    matrix: Matrix = quaternion_to_matrix(rotation)
    # column-vector layout: element (row, col) is the field named m(row + 4 * col)
    sin_pitch: float = max(-1.0, min(1.0, -matrix.m9))
    pitch: float = asin(sin_pitch)
    if abs(sin_pitch) > pitch_gimbal_limit:
        # vertical pitch: fold the degenerate yaw/roll pair into yaw alone
        sign: float = 1.0 if sin_pitch > 0.0 else -1.0
        return atan2(sign * matrix.m4, matrix.m0), pitch, 0.0
    return (
        atan2(matrix.m8, matrix.m10),
        pitch,
        atan2(matrix.m1, matrix.m5),
    )
