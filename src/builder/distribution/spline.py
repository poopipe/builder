"""cubic bezier spline sampling — returns generated slots"""

from __future__ import annotations

from pyray import (
    Transform,
    Vector3,
    matrix_invert,
    matrix_multiply,
    vector3_add,
    vector3_length,
    vector3_scale,
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
from builder.scene.scene_types import Node, NodeRole, transform_at
from builder.scene.transforms import matrix_translation, world_matrix


def bezier_point(
    p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float
) -> Vector3:
    """evaluate a cubic bezier at t in [0, 1]"""
    u: float = 1.0 - t
    uu: float = u * u
    tt: float = t * t
    return vector3_add(
        vector3_add(
            vector3_scale(p0, uu * u),
            vector3_scale(p1, 3.0 * uu * t),
        ),
        vector3_add(
            vector3_scale(p2, 3.0 * u * tt),
            vector3_scale(p3, tt * t),
        ),
    )


def bezier_tangent(
    p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float
) -> Vector3:
    """evaluate cubic bezier derivative at t"""
    u: float = 1.0 - t
    return vector3_add(
        vector3_add(
            vector3_scale(vector3_subtract(p1, p0), 3.0 * u * u),
            vector3_scale(vector3_subtract(p2, p1), 6.0 * u * t),
        ),
        vector3_scale(vector3_subtract(p3, p2), 3.0 * t * t),
    )


def child_with_role(
    nodes: dict[str, Node], parent_id: str, role: NodeRole
) -> Node | None:
    """return the first direct child with the given role"""
    node: Node
    for node in nodes.values():
        if node.parent_id == parent_id and node.role is role:
            return node
    return None


def ordered_bezier_points(nodes: dict[str, Node], group_id: str) -> list[Node]:
    """direct bezier point children, ordered by name then id"""
    points: list[Node] = [
        node
        for node in nodes.values()
        if node.parent_id == group_id and node.role is NodeRole.bezier_point
    ]
    points.sort(key=lambda node: (node.name, node.id))
    return points


def group_local_position(
    nodes: dict[str, Node], group_id: str, node_id: str
) -> Vector3:
    """return a node's world position in the generator group's local space"""
    local = matrix_multiply(
        world_matrix(nodes, node_id),
        matrix_invert(world_matrix(nodes, group_id)),
    )
    return matrix_translation(local)


def handle_local(
    nodes: dict[str, Node],
    group_id: str,
    point: Node,
    role: NodeRole,
    fallback: Vector3,
) -> Vector3:
    """return handle position in group-local space, or fallback when missing"""
    handle: Node | None = child_with_role(nodes, point.id, role)
    if handle is None:
        return fallback
    return group_local_position(nodes, group_id, handle.id)


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


def t_at_arc_length(
    cumulative: list[float],
    probe_ts: list[float],
    target: float,
) -> float:
    """return parameter t for a distance along the probed polyline"""
    total: float = cumulative[-1]
    if total <= 1e-12:
        return 0.0
    clamped: float = max(0.0, min(total, target))
    index: int
    for index in range(1, len(cumulative)):
        if cumulative[index] >= clamped:
            span: float = cumulative[index] - cumulative[index - 1]
            if span < 1e-12:
                return probe_ts[index]
            alpha: float = (clamped - cumulative[index - 1]) / span
            return probe_ts[index - 1] + (probe_ts[index] - probe_ts[index - 1]) * alpha
    return probe_ts[-1]


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
    """build stacked spline slots from bezier control children"""
    nodes: dict[str, Node] = dict(context.nodes)
    group_id: str = context.group_id
    points: list[Node] = ordered_bezier_points(nodes, group_id)
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
                nodes, group_id, point, NodeRole.bezier_handle_out, position
            )
        )
        ins.append(
            handle_local(nodes, group_id, point, NodeRole.bezier_handle_in, position)
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
