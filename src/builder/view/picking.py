"""ray picking against meshed scene nodes"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import BoundingBox, Ray, RayCollision, get_ray_collision_box

from builder.scene.bounds import transform_bounding_box
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.view.draw_cache import DrawCache, sync_draw_cache, world_matrix_for
from builder.view.mesh_table import MeshTable
from builder.view.prepare_mesh import PreparedMesh


@dataclass(frozen=True)
class MeshPickHit:
    """nearest meshed node hit by a ray"""

    node_id: str
    distance: float


def pick_nearest_mesh_node(
    scene: Scene,
    mesh_table: MeshTable,
    ray: Ray,
    cache: DrawCache,
) -> MeshPickHit | None:
    """return the nearest meshed node hit by ray, or none"""
    sync_draw_cache(scene, cache)
    best: MeshPickHit | None = None
    node: Node
    for node in scene.all_nodes():
        if node.mesh_id is None:
            continue
        if not mesh_table.has_mesh(node.mesh_id):
            continue
        prepared: PreparedMesh = mesh_table.get(node.mesh_id)
        world_bounds: BoundingBox = transform_bounding_box(
            prepared.local_bounds,
            world_matrix_for(scene, cache, node.id),
        )
        hit: RayCollision = get_ray_collision_box(ray, world_bounds)
        if not hit.hit:
            continue
        if best is None or hit.distance < best.distance:
            best = MeshPickHit(node_id=node.id, distance=float(hit.distance))
    return best
