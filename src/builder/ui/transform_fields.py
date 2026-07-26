"""values behind the type-in transform fields, read from and applied to selection

world space fields hold absolute values, so typing 0 0 0 as a position moves the
selection to the origin. local space fields hold an offset in the node's own axes,
so 0 does nothing. scale is a local property either way, so it stays absolute
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import degrees, radians

from pyray import Transform, Vector3, quaternion_from_axis_angle, vector3_scale

from builder.generators.regenerate import regenerate_splines_touching
from builder.scene.orientation import (
    quaternion_from_yaw_pitch_roll,
    yaw_pitch_roll_from_quaternion,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.transform_ops import (
    rotate_transform_local,
    set_transform_rotation,
    set_transform_scale,
    set_transform_translation,
    translate_transform_local,
)
from builder.view.gizmo_types import GizmoMode, GizmoSpace

transform_min_scale: float = 0.001
transform_axis_keys: tuple[str, str, str] = ("x", "y", "z")
transform_rotation_keys: tuple[str, str, str] = ("yaw", "pitch", "roll")


def transform_field_keys(mode: GizmoMode) -> tuple[str, str, str]:
    """field keys, in tab order, for the current gizmo tool"""
    if mode is GizmoMode.rotate:
        return transform_rotation_keys
    return transform_axis_keys


def transform_field_labels(mode: GizmoMode) -> tuple[str, str, str]:
    """field captions for the current gizmo tool"""
    if mode is GizmoMode.rotate:
        return ("Yaw", "Pitch", "Roll")
    return ("X", "Y", "Z")


def field_axis(mode: GizmoMode, index: int) -> Vector3:
    """unit axis one field drives: x/y/z to move or scale, y/x/z for yaw/pitch/roll"""
    if mode is GizmoMode.rotate:
        return (
            Vector3(0.0, 1.0, 0.0),
            Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 1.0),
        )[index]
    return (
        Vector3(1.0, 0.0, 0.0),
        Vector3(0.0, 1.0, 0.0),
        Vector3(0.0, 0.0, 1.0),
    )[index]


def selected_nodes(scene: Scene) -> list[Node]:
    """selected groups in scene order; nodes with a parent are not editable here"""
    return [
        node
        for node in scene.nodes.values()
        if node.id in scene.selected_ids and node.parent_id is None
    ]


def relative_fields(mode: GizmoMode, space: GizmoSpace) -> bool:
    """true when the fields hold an offset rather than an absolute value"""
    return space is GizmoSpace.local and mode is not GizmoMode.scale


def transform_values(
    transform: Transform,
    mode: GizmoMode,
) -> tuple[float, float, float]:
    """absolute values of one transform for the current tool"""
    if mode is GizmoMode.translate:
        return (
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        )
    if mode is GizmoMode.rotate:
        yaw: float
        pitch: float
        roll: float
        yaw, pitch, roll = yaw_pitch_roll_from_quaternion(transform.rotation)
        return (degrees(yaw), degrees(pitch), degrees(roll))
    return (transform.scale.x, transform.scale.y, transform.scale.z)


def transform_field_values(
    scene: Scene,
    mode: GizmoMode,
    space: GizmoSpace,
) -> tuple[float, float, float]:
    """values to display for the current selection"""
    if relative_fields(mode, space):
        return (0.0, 0.0, 0.0)
    nodes: list[Node] = selected_nodes(scene)
    if not nodes:
        return (0.0, 0.0, 0.0)
    return transform_values(nodes[0].transform, mode)


def with_field_value(
    values: tuple[float, float, float],
    index: int,
    value: float,
) -> tuple[float, float, float]:
    """return values with one component replaced"""
    return (
        value if index == 0 else values[0],
        value if index == 1 else values[1],
        value if index == 2 else values[2],
    )


def edited_transform(
    transform: Transform,
    mode: GizmoMode,
    space: GizmoSpace,
    index: int,
    value: float,
) -> Transform:
    """return transform with one typed field applied"""
    if relative_fields(mode, space):
        if mode is GizmoMode.translate:
            return translate_transform_local(
                transform, vector3_scale(field_axis(mode, index), value)
            )
        return rotate_transform_local(
            transform,
            quaternion_from_axis_angle(field_axis(mode, index), radians(value)),
        )

    x: float
    y: float
    z: float
    x, y, z = with_field_value(transform_values(transform, mode), index, value)
    if mode is GizmoMode.translate:
        return set_transform_translation(transform, Vector3(x, y, z))
    if mode is GizmoMode.rotate:
        return set_transform_rotation(
            transform,
            quaternion_from_yaw_pitch_roll(radians(x), radians(y), radians(z)),
        )
    return set_transform_scale(
        transform,
        Vector3(
            max(transform_min_scale, x),
            max(transform_min_scale, y),
            max(transform_min_scale, z),
        ),
    )


def apply_transform_field(
    scene: Scene,
    mode: GizmoMode,
    space: GizmoSpace,
    index: int,
    value: float,
) -> None:
    """write one typed field to every selected group"""
    touched: list[str] = []
    node: Node
    for node in selected_nodes(scene):
        scene.set_node(
            replace(
                node,
                transform=edited_transform(
                    node.transform, mode, space, index, value
                ),
            )
        )
        touched.append(node.id)
    regenerate_splines_touching(scene, touched)


def format_transform_value(value: float) -> str:
    """format a field value for display, matching the inspector"""
    return f"{value:.4g}"


def parse_transform_value(text: str) -> float | None:
    """parse a typed field value, or None if it is not a number"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def field_index(keys: Sequence[str], key: str) -> int | None:
    """position of a field key in tab order, or None when it no longer exists"""
    if key not in keys:
        return None
    return keys.index(key)
