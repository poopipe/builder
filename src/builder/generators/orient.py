"""shared orientation post-pass for generated slots"""

from __future__ import annotations

from collections.abc import Mapping
from math import radians

from pyray import Transform, quaternion_multiply

from builder.generators.generator_types import GeneratedSlot, ParamValue, SlotRole
from builder.scene.orientation import quaternion_from_yaw_pitch_roll


def orient_param_keys(role: SlotRole) -> tuple[str, str, str]:
    """return yaw/pitch/roll param keys for a slot role"""
    if role == "point":
        return ("point_orient_yaw", "point_orient_pitch", "point_orient_roll")
    return ("orient_yaw", "orient_pitch", "orient_roll")


def apply_orient_offset(
    transform: Transform,
    params: Mapping[str, ParamValue],
    role: SlotRole = "default",
) -> Transform:
    """compose local yaw/pitch/roll degrees onto a generated transform"""
    yaw_key: str
    pitch_key: str
    roll_key: str
    yaw_key, pitch_key, roll_key = orient_param_keys(role)
    yaw: float = radians(float(params.get(yaw_key, 0.0)))
    pitch: float = radians(float(params.get(pitch_key, 0.0)))
    roll: float = radians(float(params.get(roll_key, 0.0)))
    if abs(yaw) < 1e-12 and abs(pitch) < 1e-12 and abs(roll) < 1e-12:
        return transform
    return Transform(
        transform.translation,
        quaternion_multiply(
            transform.rotation, quaternion_from_yaw_pitch_roll(yaw, pitch, roll)
        ),
        transform.scale,
    )


def apply_orient_to_slots(
    slots: list[GeneratedSlot],
    params: Mapping[str, ParamValue],
) -> list[GeneratedSlot]:
    """return slots with role-appropriate orient euler applied"""
    return [
        GeneratedSlot(
            apply_orient_offset(slot.transform, params, slot.role), slot.role
        )
        for slot in slots
    ]


def slots_from_transforms(
    transforms: list[Transform],
    role: SlotRole = "default",
) -> list[GeneratedSlot]:
    """wrap plain transforms as slots with a shared role"""
    return [GeneratedSlot(transform, role) for transform in transforms]
