"""xz translate gizmo for selected road polyline vertices"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from pyray import (
    Camera3D,
    Matrix,
    Ray,
    RayCollision,
    Vector3,
    draw_cube,
    get_ray_collision_sphere,
    matrix_invert,
    vector3_add,
    vector3_cross_product,
    vector3_normalize,
    vector3_subtract,
    vector3_transform,
)

from builder.heightfield.road_sections.edit import (
    replace_usage_polylines,
    set_usage_group_on_node,
)
from builder.heightfield.road_sections.overlay import (
    find_road_sections_layer_ancestor,
    road_sections_plane_y,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.transforms import world_matrix
from builder.view.gizmo import (
    GizmoState,
    draw_gizmo,
    gizmo_axis_color,
    gizmo_size,
    hit_test_gizmo,
    plane_intersect,
    project_ray_onto_axis,
)
from builder.view.gizmo_types import GizmoAxis, GizmoMode


@dataclass(frozen=True)
class EndpointStart:
    """selected endpoint and its xz at drag start"""

    endpoint: RoadEndpointRef
    x: float
    z: float


@dataclass
class RoadVertexAxisDrag:
    """in-progress move of selected road vertices (axis or xz plane)"""

    usage_group_id: str
    heightfield_group_id: str
    # x/z = single axis; uniform = free xz on the work plane
    axis: GizmoAxis
    axis_dir: Vector3
    pivot: Vector3
    start_point: Vector3
    starts: tuple[EndpointStart, ...]
    plane_normal: Vector3 | None = None


def selection_centroid_xz(
    group: RoadUsageGroupRecipe,
    endpoints: Sequence[RoadEndpointRef],
) -> tuple[float, float] | None:
    """average hf-local xz of selected endpoints, or none if empty/invalid"""
    if not endpoints:
        return None
    total_x: float = 0.0
    total_z: float = 0.0
    count: int = 0
    endpoint: RoadEndpointRef
    for endpoint in endpoints:
        polyline: RoadPolyline | None = polyline_by_id(group, endpoint.polyline_id)
        if polyline is None:
            continue
        if endpoint.vertex_index < 0 or endpoint.vertex_index >= len(
            polyline.points
        ):
            continue
        x: float
        z: float
        x, z = polyline.points[endpoint.vertex_index]
        total_x += x
        total_z += z
        count += 1
    if count == 0:
        return None
    return total_x / float(count), total_z / float(count)


def hf_local_xz_axes(hf_world: Matrix) -> dict[GizmoAxis, Vector3]:
    """world directions for heightfield-local x and z"""
    origin: Vector3 = vector3_transform(Vector3(0.0, 0.0, 0.0), hf_world)
    x_tip: Vector3 = vector3_transform(Vector3(1.0, 0.0, 0.0), hf_world)
    z_tip: Vector3 = vector3_transform(Vector3(0.0, 0.0, 1.0), hf_world)
    return {
        GizmoAxis.x: vector3_normalize(vector3_subtract(x_tip, origin)),
        GizmoAxis.z: vector3_normalize(vector3_subtract(z_tip, origin)),
    }


def hf_work_plane_normal(axes: dict[GizmoAxis, Vector3]) -> Vector3:
    """world normal of the heightfield xz work plane"""
    return vector3_normalize(
        vector3_cross_product(axes[GizmoAxis.x], axes[GizmoAxis.z])
    )


def vertex_plane_handle_radius(size: float) -> float:
    """pick radius for the center xz-plane handle"""
    return max(0.08, size * 0.14)


def vertex_plane_handle_size(size: float) -> float:
    """cube edge length for the center xz-plane handle"""
    return size * 0.22


def selection_pivot_world(
    nodes: dict[str, Node],
    usage_group_id: str,
    endpoints: Sequence[RoadEndpointRef],
) -> tuple[Vector3, Matrix, str, float] | None:
    """world pivot at selection centroid, plus hf world/plane context"""
    usage: Node | None = nodes.get(usage_group_id)
    if usage is None or usage.road_usage_group is None:
        return None
    layer_id: str | None = find_road_sections_layer_ancestor(nodes, usage.id)
    if layer_id is None:
        return None
    layer: Node | None = nodes.get(layer_id)
    if layer is None or layer.parent_id is None:
        return None
    heightfield_group_id: str = layer.parent_id
    centroid: tuple[float, float] | None = selection_centroid_xz(
        usage.road_usage_group, endpoints
    )
    if centroid is None:
        return None
    plane_y: float = road_sections_plane_y(
        nodes, layer_id, heightfield_group_id
    )
    hf_world: Matrix = world_matrix(nodes, heightfield_group_id)
    local: Vector3 = Vector3(centroid[0], plane_y, centroid[1])
    pivot: Vector3 = vector3_transform(local, hf_world)
    return pivot, hf_world, heightfield_group_id, plane_y


def draw_road_vertex_move_gizmo(
    camera: Camera3D,
    pivot: Vector3,
    axes: dict[GizmoAxis, Vector3],
) -> None:
    """draw xz translate handles and a center plane handle at the pivot"""
    size: float = gizmo_size(camera, pivot)
    draw_gizmo(
        GizmoState(mode=GizmoMode.translate),
        pivot,
        axes,
        size,
    )
    center: float = vertex_plane_handle_size(size)
    draw_cube(
        pivot,
        center,
        center,
        center,
        gizmo_axis_color[GizmoAxis.uniform],
    )


def collect_endpoint_starts(
    group: RoadUsageGroupRecipe,
    endpoints: Sequence[RoadEndpointRef],
) -> list[EndpointStart]:
    """snapshot selected endpoint xz positions"""
    starts: list[EndpointStart] = []
    endpoint: RoadEndpointRef
    for endpoint in endpoints:
        polyline: RoadPolyline | None = polyline_by_id(group, endpoint.polyline_id)
        if polyline is None:
            continue
        if endpoint.vertex_index < 0 or endpoint.vertex_index >= len(
            polyline.points
        ):
            continue
        x: float
        z: float
        x, z = polyline.points[endpoint.vertex_index]
        starts.append(EndpointStart(endpoint=endpoint, x=x, z=z))
    return starts


def pick_road_vertex_axis_drag(
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    usage_group_id: str,
    endpoints: Sequence[RoadEndpointRef],
) -> RoadVertexAxisDrag | None:
    """start an axis or xz-plane drag when the ray hits the selection move gizmo"""
    if not endpoints:
        return None
    context: tuple[Vector3, Matrix, str, float] | None = selection_pivot_world(
        nodes, usage_group_id, endpoints
    )
    if context is None:
        return None
    pivot: Vector3
    hf_world: Matrix
    heightfield_group_id: str
    pivot, hf_world, heightfield_group_id, _plane_y = context
    axes: dict[GizmoAxis, Vector3] = hf_local_xz_axes(hf_world)
    size: float = gizmo_size(camera, pivot)
    plane_normal: Vector3 = hf_work_plane_normal(axes)
    hit: GizmoAxis | None = None
    axis_dir: Vector3 = Vector3(0.0, 0.0, 0.0)
    start_point: Vector3 | None = None
    drag_plane_normal: Vector3 | None = None

    center_hit: RayCollision = get_ray_collision_sphere(
        ray, pivot, vertex_plane_handle_radius(size)
    )
    if center_hit.hit:
        hit = GizmoAxis.uniform
        drag_plane_normal = plane_normal
        start_point = plane_intersect(ray, pivot, plane_normal)
        axis_dir = axes[GizmoAxis.x]
    else:
        hit = hit_test_gizmo(
            GizmoState(mode=GizmoMode.translate), ray, pivot, axes, size
        )
        if hit is None or hit not in axes:
            return None
        axis_dir = axes[hit]
        start_point = project_ray_onto_axis(
            ray, pivot, axis_dir, camera.position
        )
    if start_point is None:
        return None
    usage: Node | None = nodes.get(usage_group_id)
    if usage is None or usage.road_usage_group is None:
        return None
    starts: list[EndpointStart] = collect_endpoint_starts(
        usage.road_usage_group, endpoints
    )
    if not starts:
        return None
    return RoadVertexAxisDrag(
        usage_group_id=usage_group_id,
        heightfield_group_id=heightfield_group_id,
        axis=hit,
        axis_dir=axis_dir,
        pivot=pivot,
        start_point=start_point,
        starts=tuple(starts),
        plane_normal=drag_plane_normal,
    )


def world_delta_to_hf_xz(
    world_delta: Vector3,
    hf_world: Matrix,
) -> tuple[float, float]:
    """convert a world-space translation into hf-local xz"""
    inv_hf: Matrix = matrix_invert(hf_world)
    world_origin: Vector3 = vector3_transform(Vector3(0.0, 0.0, 0.0), hf_world)
    local_delta: Vector3 = vector3_subtract(
        vector3_transform(vector3_add(world_origin, world_delta), inv_hf),
        vector3_transform(world_origin, inv_hf),
    )
    return float(local_delta.x), float(local_delta.z)


def axis_delta_hf_xz(
    drag: RoadVertexAxisDrag,
    ray: Ray,
    camera_position: Vector3,
    hf_world: Matrix,
) -> tuple[float, float] | None:
    """hf-local xz delta from the current ray (axis or work plane)"""
    current: Vector3 | None
    if drag.axis is GizmoAxis.uniform:
        assert drag.plane_normal is not None
        current = plane_intersect(ray, drag.pivot, drag.plane_normal)
    else:
        current = project_ray_onto_axis(
            ray, drag.pivot, drag.axis_dir, camera_position
        )
    if current is None:
        return None
    world_delta: Vector3 = vector3_subtract(current, drag.start_point)
    dx: float
    dz: float
    dx, dz = world_delta_to_hf_xz(world_delta, hf_world)
    if drag.axis is GizmoAxis.x:
        return dx, 0.0
    if drag.axis is GizmoAxis.z:
        return 0.0, dz
    if drag.axis is GizmoAxis.uniform:
        return dx, dz
    return None


def apply_endpoint_starts_delta(
    group: RoadUsageGroupRecipe,
    starts: Sequence[EndpointStart],
    dx: float,
    dz: float,
) -> RoadUsageGroupRecipe:
    """move every start endpoint by the same xz delta"""
    by_polyline: dict[str, dict[int, tuple[float, float]]] = {}
    start: EndpointStart
    for start in starts:
        points_map: dict[int, tuple[float, float]] = by_polyline.setdefault(
            start.endpoint.polyline_id, {}
        )
        points_map[start.endpoint.vertex_index] = (
            start.x + dx,
            start.z + dz,
        )
    lines: list[RoadPolyline] = []
    polyline: RoadPolyline
    for polyline in group.polylines:
        index_map: dict[int, tuple[float, float]] | None = by_polyline.get(
            polyline.id
        )
        if index_map is None:
            lines.append(polyline)
            continue
        points: list[tuple[float, float]] = list(polyline.points)
        index: int
        xz: tuple[float, float]
        for index, xz in index_map.items():
            if 0 <= index < len(points):
                points[index] = (float(xz[0]), float(xz[1]))
        lines.append(replace(polyline, points=tuple(points)))
    return replace_usage_polylines(group, lines)


def apply_road_vertex_axis_drag(
    scene: Scene,
    drag: RoadVertexAxisDrag,
    ray: Ray,
    camera_position: Vector3,
) -> None:
    """move all selected vertices by the shared axis delta"""
    usage: Node | None = scene.nodes.get(drag.usage_group_id)
    if usage is None or usage.road_usage_group is None:
        return
    hf_world: Matrix = world_matrix(scene.nodes, drag.heightfield_group_id)
    delta: tuple[float, float] | None = axis_delta_hf_xz(
        drag, ray, camera_position, hf_world
    )
    if delta is None:
        return
    updated: RoadUsageGroupRecipe = apply_endpoint_starts_delta(
        usage.road_usage_group, drag.starts, delta[0], delta[1]
    )
    scene.add_nodes([set_usage_group_on_node(usage, updated)])


def dragged_polyline_ids(drag: RoadVertexAxisDrag) -> list[str]:
    """unique polyline ids touched by an axis drag"""
    ids: list[str] = []
    seen: set[str] = set()
    start: EndpointStart
    for start in drag.starts:
        polyline_id: str = start.endpoint.polyline_id
        if polyline_id in seen:
            continue
        seen.add(polyline_id)
        ids.append(polyline_id)
    return ids
