"""ray picking against heightfield group bounds"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import BoundingBox, Ray, RayCollision, Vector3, get_ray_collision_box

from builder.heightfield.heightfield_types import (
    Heightfield,
    HeightfieldLayout,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import layer_height_bounds
from builder.scene.bounds import transform_bounding_box
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.view.draw_cache import DrawCache, fill_world_matrix, sync_draw_cache


@dataclass(frozen=True)
class HeightfieldPickHit:
    """nearest heightfield group hit by a ray"""

    group_id: str
    distance: float


def heightfield_local_bounds(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> BoundingBox:
    """axis-aligned local bounds centered on the group origin"""
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    height_min: float
    height_max: float
    height_min, height_max = layer_height_bounds(catalog, heightfield.layers)
    half_x: float = 0.5 * layout.size_x
    half_z: float = 0.5 * layout.size_z
    return BoundingBox(
        Vector3(-half_x, height_min, -half_z),
        Vector3(half_x, height_max, half_z),
    )


def pick_nearest_heightfield(
    scene: Scene,
    catalog: HeightmapCatalog,
    ray: Ray,
    cache: DrawCache,
) -> HeightfieldPickHit | None:
    """return the nearest heightfield group hit by ray, or none"""
    sync_draw_cache(scene, cache)
    best: HeightfieldPickHit | None = None
    node_id: str
    node: Node
    for node_id, node in scene.nodes.items():
        heightfield: Heightfield | None = node.heightfield
        if heightfield is None:
            continue
        local_bounds: BoundingBox = heightfield_local_bounds(catalog, heightfield)
        world_bounds: BoundingBox = transform_bounding_box(
            local_bounds,
            fill_world_matrix(scene, cache, node_id),
        )
        hit: RayCollision = get_ray_collision_box(ray, world_bounds)
        if not hit.hit:
            continue
        if best is None or hit.distance < best.distance:
            best = HeightfieldPickHit(group_id=node_id, distance=float(hit.distance))
    return best
