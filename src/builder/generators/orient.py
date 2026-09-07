"""shared orientation helpers for generated slots"""

from __future__ import annotations

from math import radians

from pyray import Transform, quaternion_multiply

from builder.generators.generator_types import GeneratedSlot, SlotRole
from builder.generators.param_types import Float3Param
from builder.scene.orientation import quaternion_from_yaw_pitch_roll


def apply_orient_offset(transform: Transform, orient: Float3Param) -> Transform:
    """compose local yaw/pitch/roll degrees (x/y/z) onto a generated transform"""
    yaw: float = radians(orient.x)
    pitch: float = radians(orient.y)
    roll: float = radians(orient.z)
    if abs(yaw) < 1e-12 and abs(pitch) < 1e-12 and abs(roll) < 1e-12:
        return transform
    return Transform(
        transform.translation,
        quaternion_multiply(
            transform.rotation, quaternion_from_yaw_pitch_roll(yaw, pitch, roll)
        ),
        transform.scale,
    )


def apply_orients(
    slots: list[GeneratedSlot],
    *,
    orient: Float3Param,
    point_orient: Float3Param,
) -> list[GeneratedSlot]:
    """apply euler offsets; point slots use point_orient, others use orient"""
    return [
        GeneratedSlot(
            apply_orient_offset(
                slot.transform,
                point_orient if slot.role == "point" else orient,
            ),
            slot.role,
        )
        for slot in slots
    ]


def slots_from_transforms(
    transforms: list[Transform],
    role: SlotRole = "default",
) -> list[GeneratedSlot]:
    """wrap plain transforms as slots with a shared role"""
    return [GeneratedSlot(transform, role) for transform in transforms]
