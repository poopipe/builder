"""ray picking against vector point cubes on the work plane"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    BoundingBox,
    Camera3D,
    Matrix,
    Ray,
    RayCollision,
    Vector3,
    get_ray_collision_box,
    vector3_distance,
)

from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.vector.coords import (
    ordered_child_nodes,
    vector_layer_is_linked,
    vector_layer_plane_y,
    vector_point_entries,
)
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
    point_cube_size_for_line_radius,
)
from builder.scene.bounds import transform_bounding_box
from builder.scene.scene_types import Node
from builder.scene.transforms import matrix_translation, world_matrix
from builder.scene.visibility import node_effectively_visible


@dataclass(frozen=True)
class VectorPointPickHit:
    """nearest vector point cube hit by a ray"""

    point_id: str
    distance: float


def point_cube_local_bounds(
    x: float, plane_y: float, z: float, cube_size: float
) -> BoundingBox:
    """axis-aligned cube bounds in heightfield-local space"""
    half: float = float(cube_size) * 0.5
    return BoundingBox(
        Vector3(x - half, plane_y - half, z - half),
        Vector3(x + half, plane_y + half, z + half),
    )


def pick_nearest_vector_point(
    nodes: dict[str, Node],
    ray: Ray,
    camera: Camera3D,
    viewport_height_px: float,
) -> VectorPointPickHit | None:
    """return the nearest linked vector point cube under the ray, or none"""
    best: VectorPointPickHit | None = None
    node: Node
    for node in nodes.values():
        heightfield: Heightfield | None = node.heightfield
        if heightfield is None:
            continue
        if not node_effectively_visible(nodes, node.id):
            continue
        hf_world: Matrix = world_matrix(nodes, node.id)
        distance_to_hf: float = vector3_distance(
            camera.position, matrix_translation(hf_world)
        )
        line_radius: float = overlay_radius_for_camera(
            camera, distance_to_hf, viewport_height_px, overlay_line_pixels
        )
        cube_size: float = point_cube_size_for_line_radius(line_radius)
        layer: Node
        for layer in ordered_child_nodes(nodes, node.id):
            if layer.vector_layer is None:
                continue
            if not vector_layer_is_linked(heightfield, layer.id):
                continue
            if not node_effectively_visible(nodes, layer.id):
                continue
            plane_y: float = vector_layer_plane_y(nodes, layer.id, node.id)
            shape: Node
            for shape in ordered_child_nodes(nodes, layer.id):
                if shape.vector_shape is None:
                    continue
                if not node_effectively_visible(nodes, shape.id):
                    continue
                entry: tuple[str, float, float]
                for entry in vector_point_entries(nodes, shape.id, node.id):
                    point_id: str
                    px: float
                    pz: float
                    point_id, px, pz = entry
                    if not node_effectively_visible(nodes, point_id):
                        continue
                    local_bounds: BoundingBox = point_cube_local_bounds(
                        px, plane_y, pz, cube_size
                    )
                    world_bounds: BoundingBox = transform_bounding_box(
                        local_bounds, hf_world
                    )
                    hit: RayCollision = get_ray_collision_box(
                        ray, world_bounds
                    )
                    if not hit.hit:
                        continue
                    distance: float = float(hit.distance)
                    if best is None or distance < best.distance:
                        best = VectorPointPickHit(
                            point_id=point_id, distance=distance
                        )
    return best
