"""mutate spline control-point hierarchy"""

from __future__ import annotations

from dataclasses import replace

from pyray import (
    Vector3,
    vector3_add,
    vector3_length,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.generators.regenerate import regenerate_group
from builder.generators.spline import SplineParams
from builder.path.points import (
    child_with_role,
    default_handle_length,
    make_path_handle,
    make_path_point,
    ordered_path_points,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node, NodeRole

default_point_step: float = 4.0


def rename_spline_points(scene: Scene, group_id: str) -> None:
    """rename path points to Point 0..n in order"""
    points: list[Node] = ordered_path_points(scene.nodes, group_id)
    index: int
    point: Node
    for index, point in enumerate(points):
        name: str = f"Point {index}"
        if point.name != name:
            scene.set_node(replace(point, name=name))


def add_spline_point(scene: Scene, group_id: str) -> str:
    """append a path point to the spline and regenerate; return the new point id"""
    group: Node = scene.nodes[group_id]
    if group.generator is None or group.generator.kind != "spline":
        raise ValueError(f"group {group_id} is not a spline")
    params: SplineParams = group.generator.params
    closed: bool = params.closed.value
    points: list[Node] = ordered_path_points(scene.nodes, group_id)
    to_add: list[Node] = []

    step: Vector3 = Vector3(default_point_step, 0.0, 0.0)
    last_pos: Vector3 = Vector3(0.0, 0.0, 0.0)
    if len(points) >= 2:
        last_pos = points[-1].transform.translation
        prev_pos: Vector3 = points[-2].transform.translation
        delta: Vector3 = vector3_subtract(last_pos, prev_pos)
        if vector3_length(delta) >= 1e-6:
            step = vector3_scale(vector3_normalize(delta), default_point_step)
    elif len(points) == 1:
        last_pos = points[0].transform.translation

    new_pos: Vector3 = vector3_add(last_pos, step)

    if points:
        last: Node = points[-1]
        out_offset: Vector3 = vector3_scale(
            vector3_normalize(step), default_handle_length
        )
        if vector3_length(out_offset) < 1e-6:
            out_offset = Vector3(default_handle_length, 0.0, 0.0)
        if child_with_role(scene.nodes, last.id, NodeRole.path_handle_out) is None:
            to_add.append(
                make_path_handle(
                    last.id, NodeRole.path_handle_out, "Out", out_offset
                )
            )
        if closed and child_with_role(
            scene.nodes, last.id, NodeRole.path_handle_in
        ) is None:
            to_add.append(
                make_path_handle(
                    last.id,
                    NodeRole.path_handle_in,
                    "In",
                    vector3_scale(out_offset, -1.0),
                )
            )

    back: Vector3 = vector3_scale(
        vector3_normalize(vector3_subtract(last_pos, new_pos)),
        default_handle_length,
    )
    if vector3_length(back) < 1e-6:
        back = Vector3(-default_handle_length, 0.0, 0.0)
    handle_out: Vector3 | None = (
        vector3_scale(back, -1.0) if closed else None
    )
    created: list[Node] = make_path_point(
        group_id,
        f"Point {len(points)}",
        new_pos,
        handle_in=back,
        handle_out=handle_out,
    )
    to_add.extend(created)
    point_id: str = created[0].id

    scene.add_nodes(to_add)
    rename_spline_points(scene, group_id)
    regenerate_group(scene, group_id)
    return point_id
