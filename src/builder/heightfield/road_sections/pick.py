"""ray picking against road sections polylines and endpoints"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from pyray import Camera3D, Matrix, Ray, Vector3, vector3_distance

from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.road_sections.overlay import (
    find_road_sections_layer_ancestor,
    road_sections_layer_is_linked,
    road_sections_plane_y,
    selected_road_sections_layer_id,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
)
from builder.heightfield.vector.circle_handle import ray_hit_hf_plane_xz
from builder.heightfield.vector.coords import ordered_child_nodes
from builder.heightfield.road_sections.vertex_handle import vertex_pick_radius
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
)
from builder.scene.scene_types import Node
from builder.scene.transforms import matrix_translation, world_matrix
from builder.scene.visibility import node_effectively_visible


@dataclass(frozen=True)
class RoadPolylinePickHit:
    """nearest road polyline under a work-plane pick"""

    usage_group_id: str
    polyline_id: str
    distance: float
    endpoint: RoadEndpointRef | None


def segment_distance_xz(
    px: float,
    pz: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> float:
    """distance from point to segment in xz"""
    dx: float = bx - ax
    dz: float = bz - az
    length_sq: float = dx * dx + dz * dz
    if length_sq < 1e-12:
        return hypot(px - ax, pz - az)
    t: float = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length_sq))
    return hypot(px - (ax + t * dx), pz - (az + t * dz))


def polyline_distance_xz(
    px: float, pz: float, polyline: RoadPolyline
) -> float:
    """min distance from point to any segment of the polyline"""
    best: float = float("inf")
    index: int
    for index in range(len(polyline.points) - 1):
        ax: float
        az: float
        ax, az = polyline.points[index]
        bx: float
        bz: float
        bx, bz = polyline.points[index + 1]
        dist: float = segment_distance_xz(px, pz, ax, az, bx, bz)
        if dist < best:
            best = dist
    return best


def nearest_vertex_ref(
    px: float,
    pz: float,
    polyline: RoadPolyline,
    vertex_radius: float,
) -> RoadEndpointRef | None:
    """polyline vertex under the pick if within radius"""
    if len(polyline.points) < 1:
        return None
    best: RoadEndpointRef | None = None
    best_dist: float = vertex_radius
    vertex_index: int
    for vertex_index, (x, z) in enumerate(polyline.points):
        dist: float = hypot(px - x, pz - z)
        if dist <= best_dist:
            best_dist = dist
            best = RoadEndpointRef(
                polyline_id=polyline.id, vertex_index=vertex_index
            )
    return best


def nearest_endpoint_ref(
    px: float,
    pz: float,
    polyline: RoadPolyline,
    endpoint_radius: float,
) -> RoadEndpointRef | None:
    """end vertex under the pick if within radius"""
    if len(polyline.points) < 2:
        return None
    ends: tuple[int, int] = (0, len(polyline.points) - 1)
    best: RoadEndpointRef | None = None
    best_dist: float = endpoint_radius
    end_index: int
    for end_index in ends:
        x: float
        z: float
        x, z = polyline.points[end_index]
        dist: float = hypot(px - x, pz - z)
        if dist <= best_dist:
            best_dist = dist
            best = RoadEndpointRef(
                polyline_id=polyline.id, vertex_index=end_index
            )
    return best


def iter_linked_usage_groups(
    nodes: dict[str, Node],
) -> list[tuple[Node, Node, RoadUsageGroupRecipe, float]]:
    """(heightfield, usage_node, group, plane_y) for linked road usage groups"""
    rows: list[tuple[Node, Node, RoadUsageGroupRecipe, float]] = []
    hf_node: Node
    for hf_node in nodes.values():
        heightfield: Heightfield | None = hf_node.heightfield
        if heightfield is None:
            continue
        layer: Node
        for layer in ordered_child_nodes(nodes, hf_node.id):
            if layer.road_sections_layer is None:
                continue
            if not road_sections_layer_is_linked(heightfield, layer.id):
                continue
            plane_y: float = road_sections_plane_y(nodes, layer.id, hf_node.id)
            shp: Node
            for shp in ordered_child_nodes(nodes, layer.id):
                if shp.road_shp is None:
                    continue
                usage: Node
                for usage in ordered_child_nodes(nodes, shp.id):
                    group: RoadUsageGroupRecipe | None = usage.road_usage_group
                    if group is None:
                        continue
                    rows.append((hf_node, usage, group, plane_y))
    return rows


# shared split verts sit at the same xz; break ties with body distance
_vertex_tie_eps: float = 1e-4


def vertex_hit_is_better(
    vert_xz: float,
    body_xz: float,
    best_vert_xz: float,
    best_body_xz: float,
) -> bool:
    """true when this vertex hit should replace the current best

    near-equal vertex distances (split join) prefer the closer polyline body
    """
    if vert_xz < best_vert_xz - _vertex_tie_eps:
        return True
    if vert_xz > best_vert_xz + _vertex_tie_eps:
        return False
    return body_xz < best_body_xz


def pick_nearest_road_polyline(
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    viewport_height_px: float,
    selected_ids: Sequence[str],
) -> RoadPolylinePickHit | None:
    """pick nearest effectively-visible road polyline under the active layer

    vertex hits win over bare segments so cubes are selectable
    """
    best_vertex: RoadPolylinePickHit | None = None
    best_vertex_xz: float = float("inf")
    best_vertex_body_xz: float = float("inf")
    best_segment: RoadPolylinePickHit | None = None
    # cache active road-sections layer id per heightfield ("" = none)
    active_layers: dict[str, str] = {}
    hf_node: Node
    usage: Node
    group: RoadUsageGroupRecipe
    plane_y: float
    for hf_node, usage, group, plane_y in iter_linked_usage_groups(nodes):
        if not node_effectively_visible(nodes, hf_node.id):
            continue
        if not node_effectively_visible(nodes, usage.id):
            continue
        cached_layer: str | None = active_layers.get(hf_node.id)
        if cached_layer is None:
            layer_id: str | None = selected_road_sections_layer_id(
                nodes, hf_node.id, selected_ids
            )
            if layer_id is None:
                active_layers[hf_node.id] = ""
                continue
            active_layers[hf_node.id] = layer_id
            cached_layer = layer_id
        if cached_layer == "":
            continue
        usage_layer_id: str | None = find_road_sections_layer_ancestor(
            nodes, usage.id
        )
        if usage_layer_id != cached_layer:
            continue
        hf_world: Matrix = world_matrix(nodes, hf_node.id)
        hit_xz: tuple[float, float] | None = ray_hit_hf_plane_xz(
            ray, hf_world, plane_y
        )
        if hit_xz is None:
            continue
        px: float
        pz: float
        px, pz = hit_xz
        distance_to_hf: float = vector3_distance(
            camera.position, matrix_translation(hf_world)
        )
        line_radius: float = overlay_radius_for_camera(
            camera, distance_to_hf, viewport_height_px, overlay_line_pixels
        )
        pick_radius: float = max(line_radius * 3.0, 1.0)
        vertex_radius: float = vertex_pick_radius(line_radius)
        translation: Vector3 = matrix_translation(hf_world)
        ray_dist: float = vector3_distance(
            camera.position,
            Vector3(
                translation.x + px,
                translation.y + plane_y,
                translation.z + pz,
            ),
        )
        polyline: RoadPolyline
        for polyline in group.polylines:
            dist_xz: float = polyline_distance_xz(px, pz, polyline)
            vertex: RoadEndpointRef | None = nearest_vertex_ref(
                px, pz, polyline, vertex_radius
            )
            if vertex is not None:
                vx: float
                vz: float
                vx, vz = polyline.points[vertex.vertex_index]
                vert_xz: float = hypot(px - vx, pz - vz)
                if vertex_hit_is_better(
                    vert_xz, dist_xz, best_vertex_xz, best_vertex_body_xz
                ):
                    best_vertex_xz = vert_xz
                    best_vertex_body_xz = dist_xz
                    best_vertex = RoadPolylinePickHit(
                        usage_group_id=usage.id,
                        polyline_id=polyline.id,
                        distance=ray_dist + vert_xz,
                        endpoint=vertex,
                    )
            if dist_xz > pick_radius:
                continue
            rank: float = ray_dist + dist_xz
            if best_segment is None or rank < best_segment.distance:
                best_segment = RoadPolylinePickHit(
                    usage_group_id=usage.id,
                    polyline_id=polyline.id,
                    distance=rank,
                    endpoint=vertex,
                )
    if best_vertex is not None:
        return best_vertex
    return best_segment


def clear_road_polyline_selection_if_stale(
    nodes: dict[str, Node],
    usage_group_id: str | None,
    polyline_ids: Sequence[str],
    selected_ids: Sequence[str],
) -> bool:
    """true when stored polyline selection no longer matches scene selection"""
    if usage_group_id is None:
        return len(polyline_ids) > 0
    if len(selected_ids) != 1 or selected_ids[0] != usage_group_id:
        return True
    node: Node | None = nodes.get(usage_group_id)
    if node is None or node.road_usage_group is None:
        return True
    known: set[str] = {polyline.id for polyline in node.road_usage_group.polylines}
    polyline_id: str
    for polyline_id in polyline_ids:
        if polyline_id not in known:
            return True
    return False
