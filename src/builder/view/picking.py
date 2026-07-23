"""Ray picking against meshed scene nodes."""

from __future__ import annotations

from pyray import BoundingBox, Ray, RayCollision, get_ray_collision_box

from builder.scene.bounds import transform_bounding_box
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.transforms import world_matrix
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


def pick_nearest_mesh_node(
    scene: Scene,
    mesh_table: MeshTable,
    ray: Ray,
) -> str | None:
    """ return id of the nearest meshed node hit by ray, or none """
    best_id: str | None = None
    best_distance: float = float("inf")
    node: Node
    for node in scene.all_nodes():
        if node.mesh_id is None:
            continue
        if not mesh_table.has_mesh(node.mesh_id):
            continue
        prepared: PreparedMesh = mesh_table.get(node.mesh_id)
        world_bounds: BoundingBox = transform_bounding_box(
            prepared.local_bounds,
            world_matrix(scene.nodes, node.id),
        )
        hit: RayCollision = get_ray_collision_box(ray, world_bounds)
        if hit.hit and hit.distance < best_distance:
            best_distance = hit.distance
            best_id = node.id
    return best_id
