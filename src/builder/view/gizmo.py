"""transform gizmo: hit-test, drag, and draw"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, pi, sin

from pyray import (
    Camera3D,
    Color,
    Ray,
    RayCollision,
    Transform,
    Vector3,
    draw_cube,
    draw_cylinder_ex,
    get_ray_collision_sphere,
    quaternion_from_axis_angle,
    quaternion_identity,
    vector3_add,
    vector3_cross_product,
    vector3_distance,
    vector3_dot_product,
    vector3_length,
    vector3_normalize,
    vector3_rotate_by_quaternion,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene import Scene
from builder.scene.scene_types import Node, Quaternion
from builder.scene.transform_ops import (
    rotate_transform_local,
    rotate_transform_world,
    scale_transform_uniform,
    translate_transform,
)
from builder.scene.transforms import matrix_translation, world_matrix
from builder.view.gizmo_types import GizmoAxis, GizmoMode, GizmoSpace

gizmo_axis_color: dict[GizmoAxis, Color] = {
    GizmoAxis.x: Color(220, 70, 70, 255),
    GizmoAxis.y: Color(70, 200, 70, 255),
    GizmoAxis.z: Color(70, 120, 220, 255),
    GizmoAxis.uniform: Color(220, 220, 80, 255),
}

gizmo_axis_radius_factor: float = 0.04
gizmo_pick_radius_factor: float = 0.08
gizmo_cylinder_sides: int = 12
gizmo_ring_segments: int = 48


@dataclass
class GizmoDrag:
    """in-progress gizmo manipulation"""

    axis: GizmoAxis
    start_transforms: dict[str, Transform]
    pivot: Vector3
    axis_dir: Vector3
    start_point: Vector3
    start_angle: float = 0.0
    start_scale_distance: float = 1.0


@dataclass
class GizmoState:
    """active gizmo tool and optional drag"""

    mode: GizmoMode = GizmoMode.translate
    space: GizmoSpace = GizmoSpace.world
    drag: GizmoDrag | None = None


def node_world_position(nodes: dict[str, Node], node_id: str) -> Vector3:
    """return world-space origin of a node"""
    return matrix_translation(world_matrix(nodes, node_id))


def selection_pivot(scene: Scene, selected_ids: set[str]) -> Vector3 | None:
    """return average world origin of selected groups, or none"""
    if not selected_ids:
        return None
    total: Vector3 = Vector3(0.0, 0.0, 0.0)
    count: int = 0
    group_id: str
    for group_id in selected_ids:
        if group_id not in scene.nodes:
            continue
        total = vector3_add(total, node_world_position(scene.nodes, group_id))
        count += 1
    if count == 0:
        return None
    return vector3_scale(total, 1.0 / float(count))


def selection_rotation(scene: Scene, selected_ids: set[str]) -> Quaternion:
    """return local rotation of the first selected group (identity if none)"""
    group_id: str
    for group_id in selected_ids:
        node: Node | None = scene.nodes.get(group_id)
        if node is not None:
            return node.transform.rotation
    return quaternion_identity()


def gizmo_axis_directions(
    space: GizmoSpace,
    local_rotation: Quaternion,
) -> dict[GizmoAxis, Vector3]:
    """return world directions for x/y/z gizmo axes"""
    local_axes: dict[GizmoAxis, Vector3] = {
        GizmoAxis.x: Vector3(1.0, 0.0, 0.0),
        GizmoAxis.y: Vector3(0.0, 1.0, 0.0),
        GizmoAxis.z: Vector3(0.0, 0.0, 1.0),
    }
    if space is GizmoSpace.world:
        return local_axes
    return {
        axis: vector3_normalize(vector3_rotate_by_quaternion(direction, local_rotation))
        for axis, direction in local_axes.items()
    }


def gizmo_size(camera: Camera3D, pivot: Vector3) -> float:
    """return gizmo length scaled by camera distance"""
    distance: float = vector3_distance(camera.position, pivot)
    return max(0.5, distance * 0.12)


def plane_intersect(
    ray: Ray,
    plane_point: Vector3,
    plane_normal: Vector3,
) -> Vector3 | None:
    """return ray-plane intersection or none"""
    denom: float = vector3_dot_product(ray.direction, plane_normal)
    if abs(denom) < 1e-8:
        return None
    t: float = (
        vector3_dot_product(
            vector3_subtract(plane_point, ray.position),
            plane_normal,
        )
        / denom
    )
    if t < 0.0:
        return None
    return vector3_add(ray.position, vector3_scale(ray.direction, t))


def axis_drag_plane_normal(
    axis_dir: Vector3,
    camera_position: Vector3,
    pivot: Vector3,
) -> Vector3:
    """return a plane normal containing axis_dir and facing the camera"""
    axis: Vector3 = vector3_normalize(axis_dir)
    to_camera: Vector3 = vector3_subtract(camera_position, pivot)
    helper: Vector3 = vector3_cross_product(axis, to_camera)
    if vector3_length(helper) < 1e-6:
        helper = vector3_cross_product(axis, Vector3(0.0, 1.0, 0.0))
        if vector3_length(helper) < 1e-6:
            helper = vector3_cross_product(axis, Vector3(1.0, 0.0, 0.0))
    return vector3_normalize(vector3_cross_product(helper, axis))


def project_ray_onto_axis(
    ray: Ray,
    pivot: Vector3,
    axis_dir: Vector3,
    camera_position: Vector3,
) -> Vector3 | None:
    """intersect ray with the axis drag plane, then project onto the axis"""
    axis: Vector3 = vector3_normalize(axis_dir)
    plane_normal: Vector3 = axis_drag_plane_normal(axis, camera_position, pivot)
    hit: Vector3 | None = plane_intersect(ray, pivot, plane_normal)
    if hit is None:
        return None
    along: float = vector3_dot_product(vector3_subtract(hit, pivot), axis)
    return vector3_add(pivot, vector3_scale(axis, along))


def closest_point_on_ray_to_segment(
    ray: Ray,
    segment_start: Vector3,
    segment_end: Vector3,
) -> tuple[Vector3, Vector3, float]:
    """return (point on ray, point on segment, distance)"""
    ray_dir: Vector3 = vector3_normalize(ray.direction)
    seg_dir: Vector3 = vector3_subtract(segment_end, segment_start)
    seg_len: float = vector3_length(seg_dir)
    if seg_len < 1e-6:
        point: Vector3 = segment_start
        to_point: Vector3 = vector3_subtract(point, ray.position)
        t: float = max(0.0, vector3_dot_product(to_point, ray_dir))
        on_ray: Vector3 = vector3_add(ray.position, vector3_scale(ray_dir, t))
        return on_ray, point, vector3_distance(on_ray, point)

    seg_unit: Vector3 = vector3_scale(seg_dir, 1.0 / seg_len)
    w0: Vector3 = vector3_subtract(ray.position, segment_start)
    a: float = vector3_dot_product(ray_dir, ray_dir)
    b: float = vector3_dot_product(ray_dir, seg_unit)
    c: float = vector3_dot_product(seg_unit, seg_unit)
    d: float = vector3_dot_product(ray_dir, w0)
    e: float = vector3_dot_product(seg_unit, w0)
    denom: float = a * c - b * b
    ray_t: float
    seg_t: float
    if abs(denom) < 1e-8:
        ray_t = 0.0
        seg_t = e / c if c > 1e-8 else 0.0
    else:
        ray_t = (b * e - c * d) / denom
        seg_t = (a * e - b * d) / denom
    ray_t = max(0.0, ray_t)
    seg_t = max(0.0, min(seg_len, seg_t))
    on_ray: Vector3 = vector3_add(ray.position, vector3_scale(ray_dir, ray_t))
    on_seg: Vector3 = vector3_add(segment_start, vector3_scale(seg_unit, seg_t))
    return on_ray, on_seg, vector3_distance(on_ray, on_seg)


def angle_about_axis(point: Vector3, pivot: Vector3, axis: Vector3) -> float:
    """return signed angle of point around axis using a stable basis"""
    offset: Vector3 = vector3_subtract(point, pivot)
    axis_n: Vector3 = vector3_normalize(axis)
    projected: Vector3 = vector3_subtract(
        offset,
        vector3_scale(axis_n, vector3_dot_product(offset, axis_n)),
    )
    if vector3_length(projected) < 1e-8:
        return 0.0
    helper: Vector3 = Vector3(0.0, 1.0, 0.0)
    if abs(vector3_dot_product(axis_n, helper)) > 0.9:
        helper = Vector3(1.0, 0.0, 0.0)
    tangent: Vector3 = vector3_normalize(vector3_cross_product(axis_n, helper))
    bitangent: Vector3 = vector3_normalize(vector3_cross_product(axis_n, tangent))
    x: float = vector3_dot_product(projected, tangent)
    y: float = vector3_dot_product(projected, bitangent)
    return atan2(y, x)


def ring_points(
    center: Vector3,
    axis: Vector3,
    radius: float,
    segments: int,
) -> list[Vector3]:
    """return world points around a rotation ring"""
    axis_n: Vector3 = vector3_normalize(axis)
    helper: Vector3 = Vector3(0.0, 1.0, 0.0)
    if abs(vector3_dot_product(axis_n, helper)) > 0.9:
        helper = Vector3(1.0, 0.0, 0.0)
    tangent: Vector3 = vector3_normalize(vector3_cross_product(axis_n, helper))
    bitangent: Vector3 = vector3_normalize(vector3_cross_product(axis_n, tangent))
    points: list[Vector3] = []
    index: int
    for index in range(segments):
        angle: float = (2.0 * pi) * (index / segments)
        points.append(
            vector3_add(
                center,
                vector3_add(
                    vector3_scale(tangent, cos(angle) * radius),
                    vector3_scale(bitangent, sin(angle) * radius),
                ),
            )
        )
    return points


def hit_test_gizmo(
    state: GizmoState,
    ray: Ray,
    pivot: Vector3,
    axes: dict[GizmoAxis, Vector3],
    size: float,
) -> GizmoAxis | None:
    """return the nearest gizmo handle under the ray in world space"""
    pick_radius: float = max(0.05, size * gizmo_pick_radius_factor)
    tip_radius: float = max(0.08, size * 0.14)
    best_axis: GizmoAxis | None = None
    best_distance: float = float("inf")
    ray_distance: float

    if state.mode is GizmoMode.scale:
        center_hit: RayCollision = get_ray_collision_sphere(ray, pivot, tip_radius)
        if center_hit.hit:
            return GizmoAxis.uniform

    axis: GizmoAxis
    direction: Vector3
    for axis, direction in axes.items():
        tip: Vector3 = vector3_add(pivot, vector3_scale(direction, size))

        if state.mode is GizmoMode.translate or state.mode is GizmoMode.scale:
            tip_hit: RayCollision = get_ray_collision_sphere(ray, tip, tip_radius)
            if tip_hit.hit and tip_hit.distance < best_distance:
                best_distance = tip_hit.distance
                best_axis = GizmoAxis.uniform if state.mode is GizmoMode.scale else axis

            on_ray: Vector3
            radial: float
            on_ray, _, radial = closest_point_on_ray_to_segment(ray, pivot, tip)
            if radial <= pick_radius:
                ray_distance = vector3_distance(ray.position, on_ray)
                if ray_distance < best_distance:
                    best_distance = ray_distance
                    best_axis = (
                        GizmoAxis.uniform if state.mode is GizmoMode.scale else axis
                    )

        elif state.mode is GizmoMode.rotate:
            points: list[Vector3] = ring_points(
                pivot, direction, size, gizmo_ring_segments
            )
            index: int
            for index in range(len(points)):
                on_ray, _, radial = closest_point_on_ray_to_segment(
                    ray,
                    points[index],
                    points[(index + 1) % len(points)],
                )
                if radial <= pick_radius:
                    ray_distance = vector3_distance(ray.position, on_ray)
                    if ray_distance < best_distance:
                        best_distance = ray_distance
                        best_axis = axis
    return best_axis


def capture_transforms(scene: Scene, selected_ids: set[str]) -> dict[str, Transform]:
    """snapshot local transforms for selected groups"""
    result: dict[str, Transform] = {}
    group_id: str
    for group_id in selected_ids:
        node: Node | None = scene.nodes.get(group_id)
        if node is None:
            continue
        result[group_id] = node.transform
    return result


def begin_gizmo_drag(
    state: GizmoState,
    scene: Scene,
    selected_ids: set[str],
    ray: Ray,
    axis: GizmoAxis,
    pivot: Vector3,
    axes: dict[GizmoAxis, Vector3],
    camera_position: Vector3,
) -> None:
    """start a gizmo drag for the given handle"""
    axis_dir: Vector3
    if axis is GizmoAxis.uniform:
        axis_dir = Vector3(1.0, 0.0, 0.0)
    else:
        axis_dir = axes[axis]

    start_point: Vector3 = pivot
    start_angle: float = 0.0
    start_scale_distance: float = 1.0

    if state.mode is GizmoMode.translate and axis is not GizmoAxis.uniform:
        projected: Vector3 | None = project_ray_onto_axis(
            ray, pivot, axis_dir, camera_position
        )
        if projected is not None:
            start_point = projected
    elif state.mode is GizmoMode.rotate and axis is not GizmoAxis.uniform:
        point: Vector3 | None = plane_intersect(ray, pivot, axis_dir)
        if point is not None:
            start_point = point
            start_angle = angle_about_axis(point, pivot, axis_dir)
    elif state.mode is GizmoMode.scale:
        plane_n: Vector3 = axis_drag_plane_normal(axis_dir, camera_position, pivot)
        point = plane_intersect(ray, pivot, plane_n)
        if point is None:
            point = pivot
        start_point = point
        start_scale_distance = max(1e-3, vector3_distance(point, pivot))

    state.drag = GizmoDrag(
        axis=axis,
        start_transforms=capture_transforms(scene, selected_ids),
        pivot=pivot,
        axis_dir=axis_dir,
        start_point=start_point,
        start_angle=start_angle,
        start_scale_distance=start_scale_distance,
    )


def apply_drag_to_scene(
    scene: Scene,
    state: GizmoState,
    ray: Ray,
    camera_position: Vector3,
) -> None:
    """update selected group transforms from the current drag"""
    drag: GizmoDrag | None = state.drag
    if drag is None:
        return

    group_id: str
    start_transform: Transform

    if state.mode is GizmoMode.translate and drag.axis is not GizmoAxis.uniform:
        current: Vector3 | None = project_ray_onto_axis(
            ray, drag.pivot, drag.axis_dir, camera_position
        )
        if current is None:
            return
        delta: Vector3 = vector3_subtract(current, drag.start_point)
        # keep motion exactly on the axis
        axis: Vector3 = vector3_normalize(drag.axis_dir)
        delta = vector3_scale(axis, vector3_dot_product(delta, axis))
        for group_id, start_transform in drag.start_transforms.items():
            scene.set_node(
                replace(
                    scene.nodes[group_id],
                    transform=translate_transform(start_transform, delta),
                )
            )
        return

    if state.mode is GizmoMode.rotate and drag.axis is not GizmoAxis.uniform:
        point: Vector3 | None = plane_intersect(ray, drag.pivot, drag.axis_dir)
        if point is None:
            return
        angle: float = angle_about_axis(point, drag.pivot, drag.axis_dir)
        delta_angle: float = angle - drag.start_angle
        rotation: Quaternion = quaternion_from_axis_angle(drag.axis_dir, delta_angle)
        for group_id, start_transform in drag.start_transforms.items():
            new_transform: Transform
            if state.space is GizmoSpace.world:
                new_transform = rotate_transform_world(
                    start_transform,
                    rotation,
                    drag.pivot,
                )
            else:
                local_axis: Vector3 = {
                    GizmoAxis.x: Vector3(1.0, 0.0, 0.0),
                    GizmoAxis.y: Vector3(0.0, 1.0, 0.0),
                    GizmoAxis.z: Vector3(0.0, 0.0, 1.0),
                }[drag.axis]
                new_transform = rotate_transform_local(
                    start_transform,
                    quaternion_from_axis_angle(local_axis, delta_angle),
                )
            scene.set_node(
                replace(
                    scene.nodes[group_id],
                    transform=new_transform,
                )
            )
        return

    if state.mode is GizmoMode.scale:
        plane_n: Vector3 = axis_drag_plane_normal(
            drag.axis_dir, camera_position, drag.pivot
        )
        point = plane_intersect(ray, drag.pivot, plane_n)
        if point is None:
            return
        distance: float = max(1e-3, vector3_distance(point, drag.pivot))
        factor: float = distance / drag.start_scale_distance
        factor = max(0.05, min(20.0, factor))
        for group_id, start_transform in drag.start_transforms.items():
            scene.set_node(
                replace(
                    scene.nodes[group_id],
                    transform=scale_transform_uniform(
                        start_transform, factor, drag.pivot
                    ),
                )
            )


def end_gizmo_drag(state: GizmoState) -> None:
    """clear the active drag"""
    state.drag = None


def draw_axis_cylinder(
    start: Vector3,
    end: Vector3,
    radius: float,
    color: Color,
) -> None:
    """draw a thick axis segment as a cylinder"""
    draw_cylinder_ex(
        start,
        end,
        radius,
        radius,
        gizmo_cylinder_sides,
        color,
    )


def draw_ring(
    center: Vector3,
    axis: Vector3,
    radius: float,
    thickness: float,
    color: Color,
) -> None:
    """draw a thick circle in the plane perpendicular to axis"""
    points: list[Vector3] = ring_points(center, axis, radius, gizmo_ring_segments)
    index: int
    for index in range(len(points)):
        draw_axis_cylinder(
            points[index],
            points[(index + 1) % len(points)],
            thickness,
            color,
        )


def draw_gizmo(
    state: GizmoState,
    pivot: Vector3,
    axes: dict[GizmoAxis, Vector3],
    size: float,
) -> None:
    """draw the active gizmo at pivot"""
    radius: float = max(0.025, size * gizmo_axis_radius_factor)
    tip_size: float = size * 0.16
    axis: GizmoAxis
    direction: Vector3
    for axis, direction in axes.items():
        color: Color = gizmo_axis_color[axis]
        tip: Vector3 = vector3_add(pivot, vector3_scale(direction, size))
        if state.mode is GizmoMode.translate:
            draw_axis_cylinder(pivot, tip, radius, color)
            draw_cube(tip, tip_size, tip_size, tip_size, color)
        elif state.mode is GizmoMode.rotate:
            draw_ring(pivot, direction, size, radius, color)
        elif state.mode is GizmoMode.scale:
            draw_axis_cylinder(pivot, tip, radius, color)
            draw_cube(tip, tip_size, tip_size, tip_size, color)
    if state.mode is GizmoMode.scale:
        center_size: float = size * 0.2
        draw_cube(
            pivot,
            center_size,
            center_size,
            center_size,
            gizmo_axis_color[GizmoAxis.uniform],
        )
