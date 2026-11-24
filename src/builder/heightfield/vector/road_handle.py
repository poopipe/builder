"""shadow-point vertical handle for road vector point height offsets"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyray import (
    BoundingBox,
    Camera3D,
    Matrix,
    Ray,
    RayCollision,
    Transform,
    Vector3,
    get_ray_collision_box,
    matrix_invert,
    vector3_distance,
    vector3_transform,
)

from builder.heightfield.heightfield_types import Heightfield, heightfield_layout
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height_below
from builder.heightfield.vector.coords import (
    node_hf_local_position,
    vector_layer_is_road,
    vector_point_belongs_to_road,
    vector_road_layer_index,
)
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
    point_cube_size_for_line_radius,
)
from builder.heightfield.vector.rasterize import (
    find_heightfield_group_for_descendant,
    find_vector_layer_ancestor,
)
from builder.heightfield.vector.vector_types import world_xz_to_uv
from builder.scene.bounds import transform_bounding_box
from builder.scene.scene import Scene
from builder.scene.scene_types import Node, NodeRole
from builder.scene.transforms import matrix_translation, world_matrix

@dataclass
class RoadHeightDrag:
    """in-progress vertical drag of a road point height handle"""

    point_id: str
    heightfield_group_id: str
    layer_id: str
    local_x: float
    local_z: float
    start_offset_y: float
    start_axis_y: float


def ray_hit_vertical_axis_y(
    ray: Ray,
    hf_world: Matrix,
    local_x: float,
    local_z: float,
) -> float | None:
    """intersect world ray with the hf-local vertical line at xz; return local y"""
    local_from_world: Matrix = matrix_invert(hf_world)
    origin: Vector3 = vector3_transform(ray.position, local_from_world)
    tip: Vector3 = vector3_transform(
        Vector3(
            ray.position.x + ray.direction.x,
            ray.position.y + ray.direction.y,
            ray.position.z + ray.direction.z,
        ),
        local_from_world,
    )
    direction: Vector3 = Vector3(
        tip.x - origin.x, tip.y - origin.y, tip.z - origin.z
    )
    dx: float = float(direction.x)
    dz: float = float(direction.z)
    ox: float = float(origin.x) - local_x
    oz: float = float(origin.z) - local_z
    denom: float = dx * dx + dz * dz
    t: float
    if denom < 1e-12:
        t = 0.0
    else:
        t = -(ox * dx + oz * dz) / denom
    if t < 0.0:
        return None
    return float(origin.y) + float(direction.y) * t


def pick_road_height_handle(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    viewport_height_px: float,
    selected_ids: list[str],
) -> RoadHeightDrag | None:
    """if a selected road point's shadow cube is under the ray, return a drag"""
    if len(selected_ids) != 1:
        return None
    point_id: str = selected_ids[0]
    point: Node | None = nodes.get(point_id)
    if point is None or point.role is not NodeRole.path_point:
        return None
    if not vector_point_belongs_to_road(nodes, point_id):
        return None
    hf_id: str | None = find_heightfield_group_for_descendant(nodes, point_id)
    layer_id: str | None = find_vector_layer_ancestor(nodes, point_id)
    if hf_id is None or layer_id is None:
        return None
    hf_node: Node | None = nodes.get(hf_id)
    if hf_node is None or hf_node.heightfield is None:
        return None
    heightfield: Heightfield = hf_node.heightfield
    if not vector_layer_is_road(heightfield, layer_id):
        return None
    layout = heightfield_layout(heightfield)
    hf_local: Vector3 = node_hf_local_position(nodes, point_id, hf_id)
    local_x: float = float(hf_local.x)
    local_z: float = float(hf_local.z)
    offset_y: float = float(point.transform.translation.y)
    su: float
    sv: float
    su, sv = world_xz_to_uv(
        local_x, local_z, layout.size_x, layout.size_z
    )
    road_index: int | None = vector_road_layer_index(heightfield, layer_id)
    if road_index is None:
        return None
    hf_world: Matrix = world_matrix(nodes, hf_id)
    distance: float = vector3_distance(
        camera.position, matrix_translation(hf_world)
    )
    shadow_y: float = sample_height_below(
        catalog, heightfield, road_index, su, sv
    )
    cube_size: float = point_cube_size_for_line_radius(
        overlay_radius_for_camera(
            camera, distance, viewport_height_px, overlay_line_pixels
        )
    )
    half: float = cube_size * 0.5
    local_bounds: BoundingBox = BoundingBox(
        Vector3(local_x - half, shadow_y - half, local_z - half),
        Vector3(local_x + half, shadow_y + half, local_z + half),
    )
    world_bounds: BoundingBox = transform_bounding_box(local_bounds, hf_world)
    hit: RayCollision = get_ray_collision_box(ray, world_bounds)
    if not hit.hit:
        return None
    start_axis_y: float | None = ray_hit_vertical_axis_y(
        ray, hf_world, local_x, local_z
    )
    if start_axis_y is None:
        start_axis_y = shadow_y
    return RoadHeightDrag(
        point_id=point_id,
        heightfield_group_id=hf_id,
        layer_id=layer_id,
        local_x=local_x,
        local_z=local_z,
        start_offset_y=offset_y,
        start_axis_y=float(start_axis_y),
    )


def apply_road_height_drag(
    scene: Scene,
    drag: RoadHeightDrag,
    ray: Ray,
) -> None:
    """update the road point local y from a vertical axis drag"""
    point: Node | None = scene.nodes.get(drag.point_id)
    if point is None or point.role is not NodeRole.path_point:
        return
    hf_world: Matrix = world_matrix(scene.nodes, drag.heightfield_group_id)
    axis_y: float | None = ray_hit_vertical_axis_y(
        ray, hf_world, drag.local_x, drag.local_z
    )
    if axis_y is None:
        return
    delta_y: float = float(axis_y) - drag.start_axis_y
    new_offset: float = drag.start_offset_y + delta_y
    translation: Vector3 = point.transform.translation
    scene.set_node(
        replace(
            point,
            transform=Transform(
                Vector3(float(translation.x), new_offset, float(translation.z)),
                point.transform.rotation,
                point.transform.scale,
            ),
        )
    )
