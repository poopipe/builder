"""virtual rim handle for editing circle shape radius"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyray import (
    BoundingBox,
    Camera3D,
    Matrix,
    Ray,
    RayCollision,
    Vector3,
    get_ray_collision_box,
    matrix_invert,
    vector3_distance,
    vector3_transform,
)

from builder.heightfield.vector.circle import (
    CircleShapeParams,
    circle_radius_from_rim,
    circle_rim_local_xz,
)
from builder.heightfield.vector.coords import (
    vector_layer_plane_y,
    vector_point_xz,
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
from builder.scene.bounds import transform_bounding_box
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.transforms import matrix_translation, world_matrix


@dataclass
class CircleRadiusDrag:
    """in-progress drag of a circle radius rim handle"""

    shape_id: str
    heightfield_group_id: str
    layer_id: str
    center_x: float
    center_z: float
    plane_y: float


def ray_hit_hf_plane_xz(
    ray: Ray,
    hf_world: Matrix,
    plane_y: float,
) -> tuple[float, float] | None:
    """intersect world ray with hf-local y=plane_y; return local xz"""
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
    if abs(float(direction.y)) < 1e-8:
        return None
    t: float = (plane_y - float(origin.y)) / float(direction.y)
    if t < 0.0:
        return None
    return (
        float(origin.x) + float(direction.x) * t,
        float(origin.z) + float(direction.z) * t,
    )


def pick_circle_radius_handle(
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    viewport_height_px: float,
    selected_ids: list[str],
) -> CircleRadiusDrag | None:
    """if the selected circle's rim cube is under the ray, return a drag seed"""
    if len(selected_ids) != 1:
        return None
    selected_id: str = selected_ids[0]
    shape: Node | None = nodes.get(selected_id)
    if shape is None or shape.vector_shape is None:
        return None
    recipe = shape.vector_shape
    if recipe.kind != "circle":
        return None
    match recipe.params:
        case CircleShapeParams(radius=radius_param):
            radius: float = float(radius_param.value)
        case _:
            return None
    hf_id: str | None = find_heightfield_group_for_descendant(nodes, shape.id)
    layer_id: str | None = find_vector_layer_ancestor(nodes, shape.id)
    if hf_id is None or layer_id is None:
        return None
    points: list[tuple[float, float]] = vector_point_xz(nodes, shape.id, hf_id)
    if len(points) == 0:
        return None
    cx: float
    cz: float
    cx, cz = points[0]
    plane_y: float = vector_layer_plane_y(nodes, layer_id, hf_id)
    rim_x: float
    rim_z: float
    rim_x, rim_z = circle_rim_local_xz(cx, cz, radius)
    hf_world: Matrix = world_matrix(nodes, hf_id)
    distance: float = vector3_distance(
        camera.position, matrix_translation(hf_world)
    )
    cube_size: float = point_cube_size_for_line_radius(
        overlay_radius_for_camera(
            camera, distance, viewport_height_px, overlay_line_pixels
        )
    )
    half: float = cube_size * 0.5
    local_bounds: BoundingBox = BoundingBox(
        Vector3(rim_x - half, plane_y - half, rim_z - half),
        Vector3(rim_x + half, plane_y + half, rim_z + half),
    )
    world_bounds: BoundingBox = transform_bounding_box(local_bounds, hf_world)
    hit: RayCollision = get_ray_collision_box(ray, world_bounds)
    if not hit.hit:
        return None
    return CircleRadiusDrag(
        shape_id=shape.id,
        heightfield_group_id=hf_id,
        layer_id=layer_id,
        center_x=cx,
        center_z=cz,
        plane_y=plane_y,
    )


def apply_circle_radius_drag(
    scene: Scene,
    drag: CircleRadiusDrag,
    ray: Ray,
) -> None:
    """update the circle radius from the ray's hit on the work plane"""
    nodes: dict[str, Node] = scene.nodes
    shape: Node | None = nodes.get(drag.shape_id)
    if shape is None or shape.vector_shape is None:
        return
    recipe = shape.vector_shape
    match recipe.params:
        case CircleShapeParams() as params:
            pass
        case _:
            return
    hf_world: Matrix = world_matrix(nodes, drag.heightfield_group_id)
    hit_xz = ray_hit_hf_plane_xz(ray, hf_world, drag.plane_y)
    if hit_xz is None:
        return
    rim_x: float
    rim_z: float
    rim_x, rim_z = hit_xz
    points: list[tuple[float, float]] = vector_point_xz(
        nodes, shape.id, drag.heightfield_group_id
    )
    if len(points) == 0:
        return
    new_radius: float = circle_radius_from_rim(
        points[0][0], points[0][1], rim_x, rim_z
    )
    updated: Node = replace(
        shape,
        vector_shape=replace(
            recipe,
            params=replace(
                params, radius=replace(params.radius, value=new_radius)
            ),
        ),
    )
    scene.set_node(updated)
