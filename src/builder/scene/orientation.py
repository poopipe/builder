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
    quaternion_multiply,
    quaternion_to_matrix,
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
