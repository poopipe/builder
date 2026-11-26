"""xz drag of road polyline vertices on the work plane"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from math import hypot

from pyray import Camera3D, Matrix, Ray, vector3_distance

from builder.heightfield.road_sections.edit import (
    set_polyline_vertex_xz,
    set_usage_group_on_node,
)
from builder.heightfield.road_sections.overlay import (
    find_road_sections_layer_ancestor,
    road_sections_plane_y,
)
from builder.heightfield.road_sections.pick import nearest_vertex_ref
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
)
from builder.heightfield.vector.circle_handle import ray_hit_hf_plane_xz
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
    point_cube_size_for_line_radius,
)
from builder.heightfield.vector.work_plane_point import (
    WorkPlanePointDrag,
    begin_work_plane_point_drag,
    work_plane_point_drag_xz,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.transforms import matrix_translation, world_matrix


@dataclass
class RoadVertexDrag:
    """in-progress xz drag of a road polyline vertex"""

    usage_group_id: str
    endpoint: RoadEndpointRef
    plane: WorkPlanePointDrag


def pick_road_vertex_drag(
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    viewport_height_px: float,
    usage_group_id: str,
    polyline_ids: Sequence[str],
) -> RoadVertexDrag | None:
    """start a vertex drag when the ray hits a selected polyline vertex"""
    if not polyline_ids:
        return None
    usage: Node | None = nodes.get(usage_group_id)
    if usage is None or usage.road_usage_group is None:
        return None
    group: RoadUsageGroupRecipe = usage.road_usage_group
    layer_id: str | None = find_road_sections_layer_ancestor(nodes, usage.id)
    if layer_id is None or usage.parent_id is None:
        return None
    # heightfield is the layer's parent
    layer: Node | None = nodes.get(layer_id)
    if layer is None or layer.parent_id is None:
        return None
    heightfield_group_id: str = layer.parent_id
    plane_y: float = road_sections_plane_y(
        nodes, layer_id, heightfield_group_id
    )
    hf_world: Matrix = world_matrix(nodes, heightfield_group_id)
    hit_xz: tuple[float, float] | None = ray_hit_hf_plane_xz(
        ray, hf_world, plane_y
    )
    if hit_xz is None:
        return None
    px: float
    pz: float
    px, pz = hit_xz
    distance_to_hf: float = vector3_distance(
        camera.position, matrix_translation(hf_world)
    )
    line_radius: float = overlay_radius_for_camera(
        camera, distance_to_hf, viewport_height_px, overlay_line_pixels
    )
    vertex_radius: float = max(
        0.75,
        point_cube_size_for_line_radius(line_radius),
        max(line_radius * 3.0, 1.0),
    )
    selected: set[str] = set(polyline_ids)
    best: RoadEndpointRef | None = None
    best_dist: float = vertex_radius
    polyline_id: str
    for polyline_id in selected:
        polyline: RoadPolyline | None = polyline_by_id(group, polyline_id)
        if polyline is None:
            continue
        hit: RoadEndpointRef | None = nearest_vertex_ref(
            px, pz, polyline, vertex_radius
        )
        if hit is None:
            continue
        x: float
        z: float
        x, z = polyline.points[hit.vertex_index]
        dist: float = hypot(px - x, pz - z)
        if dist <= best_dist:
            best_dist = dist
            best = hit
    if best is None:
        return None
    polyline_hit: RoadPolyline | None = polyline_by_id(group, best.polyline_id)
    if polyline_hit is None:
        return None
    point_x: float
    point_z: float
    point_x, point_z = polyline_hit.points[best.vertex_index]
    plane: WorkPlanePointDrag | None = begin_work_plane_point_drag(
        ray,
        hf_world,
        plane_y,
        point_x,
        point_z,
        heightfield_group_id,
    )
    if plane is None:
        return None
    return RoadVertexDrag(
        usage_group_id=usage_group_id,
        endpoint=best,
        plane=plane,
    )


def apply_road_vertex_drag(scene: Scene, drag: RoadVertexDrag, ray: Ray) -> None:
    """move the dragged vertex to the current ray hit on the work plane"""
    xz: tuple[float, float] | None = work_plane_point_drag_xz(
        scene.nodes, drag.plane, ray
    )
    if xz is None:
        return
    usage: Node | None = scene.nodes.get(drag.usage_group_id)
    if usage is None or usage.road_usage_group is None:
        return
    updated: RoadUsageGroupRecipe = set_polyline_vertex_xz(
        usage.road_usage_group,
        drag.endpoint.polyline_id,
        drag.endpoint.vertex_index,
        xz[0],
        xz[1],
    )
    scene.add_nodes([set_usage_group_on_node(usage, updated)])
