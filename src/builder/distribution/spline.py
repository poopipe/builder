"""cubic bezier spline sampling — returns generated slots"""

from __future__ import annotations

from pyray import (
    Transform,
    Vector3,
    vector3_add,
    vector3_length,
    vector3_subtract,
)

from builder.distribution.grids import (
    AxisIndex,
    align_slots,
    axis_vector,
    clamp_axis,
    rotation_looking_along,
)
from builder.generators.generator_types import BuildContext, GeneratedSlot, SlotRole
from builder.path.bezier import bezier_point, bezier_tangent, t_at_arc_length
from builder.path.points import (
    group_local_position,
    handle_local,
    ordered_path_points,
)
from builder.scene.scene_types import Node, NodeRole, transform_at


def ordered_bezier_points(nodes: dict[str, Node], group_id: str) -> list[Node]:
    """alias for ordered_path_points (insertion order)"""
    return ordered_path_points(nodes, group_id)


def append_spline_slot(
    slots: list[GeneratedSlot],
    position: Vector3,
    tangent: Vector3,
    up: Vector3,
    facing: int,
    role: SlotRole,
) -> None:
    """append a spline sample with optional edge-aligned facing"""
    if facing == 2 and vector3_length(tangent) >= 1e-6:
        slots.append(
            GeneratedSlot(
                Transform(
                    position,
                    rotation_looking_along(tangent, up),
                    Vector3(1.0, 1.0, 1.0),
                ),
                role,
            )
        )
        return
    slots.append(GeneratedSlot(transform_at(position), role))


def sample_segment(
    slots: list[GeneratedSlot],
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    p3: Vector3,
    spacing: float,
    include_start: bool,
    edge_facing: int,
    point_facing: int,
    up: Vector3,
) -> None:
    """sample one cubic segment by arc length; excludes the endpoint"""
    probe_count: int = 64
    probe_ts: list[float] = [
        float(i) / float(probe_count) for i in range(probe_count + 1)
    ]
    probe: list[Vector3] = [
        bezier_point(p0, p1, p2, p3, t) for t in probe_ts
    ]
    cumulative: list[float] = [0.0]
    index: int
    for index in range(probe_count):
        cumulative.append(
            cumulative[-1]
            + vector3_length(vector3_subtract(probe[index + 1], probe[index]))
        )
    length: float = cumulative[-1]
    divisions: int = max(1, int(round(length / max(1e-6, spacing))))
    first: int = 0 if include_start else 1
    sample: int
    for sample in range(first, divisions):
        # equal arc steps; t=1 belongs to the next segment / final point
        target: float = length * float(sample) / float(divisions)
        t: float = t_at_arc_length(cumulative, probe_ts, target)
        position: Vector3 = bezier_point(p0, p1, p2, p3, t)
        tangent: Vector3 = bezier_tangent(p0, p1, p2, p3, t)
        role: SlotRole = SlotRole.point if sample == 0 else SlotRole.edge
        facing: int = point_facing if role is SlotRole.point else edge_facing
        append_spline_slot(slots, position, tangent, up, facing, role)


def spline_slots_from_context(
    context: BuildContext,
    spacing: float,
    closed: bool,
    include_points: bool,
    facing: int,
    point_facing: int,
    axis: AxisIndex,
    count_height: int,
    spacing_height: float,
    build_from: AxisIndex,
) -> list[GeneratedSlot]:
    """build stacked spline slots from path control children"""
    nodes: dict[str, Node] = dict(context.nodes)
    group_id: str = context.group_id
    points: list[Node] = ordered_path_points(nodes, group_id)
    if len(points) < 2:
        return []
    locals_points: list[Vector3] = [
        group_local_position(nodes, group_id, point.id) for point in points
    ]
    outs: list[Vector3] = []
    ins: list[Vector3] = []
    point: Node
    position: Vector3
    for point, position in zip(points, locals_points, strict=True):
        outs.append(
            handle_local(
                nodes, group_id, point, NodeRole.path_handle_out, position
            )
        )
        ins.append(
            handle_local(
                nodes, group_id, point, NodeRole.path_handle_in, position
            )
        )

    cylinder: AxisIndex = clamp_axis(axis)
    up: Vector3 = axis_vector(cylinder, 1.0)
    height_count: int = max(1, int(count_height))
    base_slots: list[GeneratedSlot] = []
    segment_count: int = len(points) if closed else len(points) - 1
    segment: int
    for segment in range(segment_count):
        i0: int = segment % len(points)
        i1: int = (segment + 1) % len(points)
        sample_segment(
            base_slots,
            locals_points[i0],
            outs[i0],
            ins[i1],
            locals_points[i1],
            spacing,
            include_points,
            facing,
            point_facing,
            up,
        )
    if include_points and not closed:
        i0 = segment_count - 1
        i1 = segment_count
        append_spline_slot(
            base_slots,
            locals_points[i1],
            bezier_tangent(
                locals_points[i0], outs[i0], ins[i1], locals_points[i1], 1.0
            ),
            up,
            point_facing,
            SlotRole.point,
        )

    slots: list[GeneratedSlot] = []
    ih: int
    for ih in range(height_count):
        height: float = (float(ih) - (height_count - 1) * 0.5) * spacing_height
        height_offset: Vector3 = axis_vector(cylinder, height)
        slot: GeneratedSlot
        for slot in base_slots:
            slots.append(
                GeneratedSlot(
                    Transform(
                        vector3_add(slot.transform.translation, height_offset),
                        slot.transform.rotation,
                        slot.transform.scale,
                    ),
                    slot.role,
                )
            )
    return align_slots(slots, Vector3(0.0, 0.0, 0.0), build_from)
