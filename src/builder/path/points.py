"""shared path-point hierarchy helpers"""

from __future__ import annotations

from pyray import (
    Vector3,
    matrix_invert,
    matrix_multiply,
    vector3_length,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.generators.param_types import FloatParam
from builder.path.bezier import (
    default_densify_step_metres,
    densify_segment_positions,
    densify_segment_xz_attrs,
)
from builder.path.path_types import PathPointRecipe
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node, NodeRole, transform_at
from builder.scene.transforms import matrix_translation, world_matrix

default_handle_length: float = 1.5


def default_path_point_recipe() -> PathPointRecipe:
    """default per-point recipe (width scale 1)"""
    return PathPointRecipe(
        width_scale=FloatParam(
            1.0, label="Width scale", step=0.05, minimum=0.05
        )
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


def ordered_path_points(nodes: dict[str, Node], parent_id: str) -> list[Node]:
    """direct path_point children in scene insertion order"""
    return [
        node
        for node in nodes.values()
        if node.parent_id == parent_id and node.role is NodeRole.path_point
    ]


def group_local_position(
    nodes: dict[str, Node], group_id: str, node_id: str
) -> Vector3:
    """return a node's world position in the group's local space"""
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


def make_path_handle(
    parent_id: str, role: NodeRole, label: str, offset: Vector3
) -> Node:
    """return a path handle child node"""
    return Node(
        id=new_node_id(),
        parent_id=parent_id,
        transform=transform_at(offset),
        mesh_id=None,
        name=label,
        role=role,
    )


def make_path_point(
    parent_id: str,
    name: str,
    position: Vector3,
    *,
    handle_in: Vector3 | None = None,
    handle_out: Vector3 | None = None,
) -> list[Node]:
    """path_point with recipe; optional in/out handles"""
    point_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=point_id,
            parent_id=parent_id,
            transform=transform_at(position),
            mesh_id=None,
            path_point=default_path_point_recipe(),
            name=name,
            role=NodeRole.path_point,
        )
    ]
    if handle_in is not None:
        nodes.append(
            make_path_handle(
                point_id, NodeRole.path_handle_in, "In", handle_in
            )
        )
    if handle_out is not None:
        nodes.append(
            make_path_handle(
                point_id, NodeRole.path_handle_out, "Out", handle_out
            )
        )
    return nodes


def path_point_width_scale(child: Node) -> float:
    """width scale from a path_point recipe; default 1 when missing"""
    if child.path_point is None:
        return 1.0
    return max(0.05, float(child.path_point.width_scale.value))


def densify_path_positions(
    nodes: dict[str, Node],
    parent_id: str,
    frame_id: str,
    *,
    closed: bool,
    step_metres: float = default_densify_step_metres,
    only_segments: set[int] | None = None,
) -> list[Vector3]:
    """densify cubic path to positions in frame-local space

    only_segments: when set, densify only those segment indices
    """
    points: list[Node] = ordered_path_points(nodes, parent_id)
    if len(points) == 0:
        return []
    locals_points: list[Vector3] = [
        group_local_position(nodes, frame_id, point.id) for point in points
    ]
    if len(points) == 1:
        return [locals_points[0]]
    outs: list[Vector3] = []
    ins: list[Vector3] = []
    point: Node
    position: Vector3
    for point, position in zip(points, locals_points, strict=True):
        outs.append(
            handle_local(
                nodes, frame_id, point, NodeRole.path_handle_out, position
            )
        )
        ins.append(
            handle_local(
                nodes, frame_id, point, NodeRole.path_handle_in, position
            )
        )
    samples: list[Vector3] = []
    segment_count: int = len(points) if closed else len(points) - 1
    selected: list[int]
    if only_segments is None:
        selected = list(range(segment_count))
    else:
        selected = sorted(
            index for index in only_segments if 0 <= index < segment_count
        )
    if len(selected) == 0:
        return []
    first_selected: int = selected[0]
    last_selected: int = selected[-1]
    segment: int
    for segment in selected:
        i0: int = segment % len(points)
        i1: int = (segment + 1) % len(points)
        samples.extend(
            densify_segment_positions(
                locals_points[i0],
                outs[i0],
                ins[i1],
                locals_points[i1],
                step_metres,
                include_start=(segment == first_selected),
                include_end=(segment == last_selected),
            )
        )
    return samples


def path_segments_near_point_ids(
    nodes: dict[str, Node],
    parent_id: str,
    point_ids: set[str],
    *,
    closed: bool,
) -> set[int]:
    """segment indices adjacent to the given path-point ids"""
    points: list[Node] = ordered_path_points(nodes, parent_id)
    if len(points) < 2 or len(point_ids) == 0:
        return set()
    segment_count: int = len(points) if closed else len(points) - 1
    near: set[int] = set()
    index: int
    for index, point in enumerate(points):
        if point.id not in point_ids:
            continue
        if closed:
            near.add((index - 1) % segment_count)
            near.add(index % segment_count)
        else:
            if index > 0:
                near.add(index - 1)
            if index < segment_count:
                near.add(index)
    return near


def densify_path_xzyw(
    nodes: dict[str, Node],
    parent_id: str,
    frame_id: str,
    *,
    closed: bool,
    step_metres: float = default_densify_step_metres,
) -> list[tuple[float, float, float, float]]:
    """densify to (x, z, parent-local y offset, width_scale)

    position xz and y use frame-local knot/handle poses; y on knots is
    parent-local translation.y so workplane height does not inflate offsets
    """
    points: list[Node] = ordered_path_points(nodes, parent_id)
    if len(points) == 0:
        return []
    # xz from frame-local; y from parent-local translation
    locals_xz: list[Vector3] = [
        group_local_position(nodes, frame_id, point.id) for point in points
    ]
    locals_points: list[Vector3] = []
    scales: list[float] = []
    point: Node
    local_xz: Vector3
    for point, local_xz in zip(points, locals_xz, strict=True):
        locals_points.append(
            Vector3(
                float(local_xz.x),
                float(point.transform.translation.y),
                float(local_xz.z),
            )
        )
        scales.append(path_point_width_scale(point))
    if len(points) == 1:
        p: Vector3 = locals_points[0]
        return [(float(p.x), float(p.z), float(p.y), scales[0])]
    outs: list[Vector3] = []
    ins: list[Vector3] = []
    for point, position in zip(points, locals_points, strict=True):
        # handles: xz from frame-local, keep knot y for cubic in xz+y-offset
        out_raw: Vector3 = handle_local(
            nodes, frame_id, point, NodeRole.path_handle_out, position
        )
        in_raw: Vector3 = handle_local(
            nodes, frame_id, point, NodeRole.path_handle_in, position
        )
        outs.append(
            Vector3(float(out_raw.x), float(position.y), float(out_raw.z))
        )
        ins.append(
            Vector3(float(in_raw.x), float(position.y), float(in_raw.z))
        )
    samples: list[tuple[float, float, float, float]] = []
    segment_count: int = len(points) if closed else len(points) - 1
    segment: int
    for segment in range(segment_count):
        i0 = segment % len(points)
        i1 = (segment + 1) % len(points)
        samples.extend(
            densify_segment_xz_attrs(
                locals_points[i0],
                outs[i0],
                ins[i1],
                locals_points[i1],
                scales[i0],
                scales[i1],
                step_metres,
                include_start=(segment == 0),
                include_end=(segment == segment_count - 1),
            )
        )
    return samples


def handle_offset_along(
    direction: Vector3, length: float = default_handle_length
) -> Vector3:
    """unit direction * length, or +x fallback"""
    if vector3_length(direction) < 1e-6:
        return Vector3(length, 0.0, 0.0)
    return vector3_scale(vector3_normalize(direction), length)


def opposite_handle(offset: Vector3) -> Vector3:
    """mirrored handle offset"""
    return vector3_scale(offset, -1.0)


def segment_direction(from_pos: Vector3, to_pos: Vector3) -> Vector3:
    """vector from from_pos to to_pos"""
    return vector3_subtract(to_pos, from_pos)
