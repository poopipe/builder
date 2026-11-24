"""scene value types: mesh ids and nodes"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from pyray import Transform, Vector3, Vector4, quaternion_identity

if TYPE_CHECKING:
    from builder.generators.generator_types import Generator
    from builder.heightfield.heightfield_types import Heightfield
    from builder.heightfield.road_sections.road_sections_types import (
        RoadSectionsLayerRecipe,
        RoadShpRecipe,
        RoadUsageGroupRecipe,
    )
    from builder.heightfield.vector.vector_types import (
        VectorLayerRecipe,
        VectorShapeRecipe,
    )
    from builder.path.path_types import PathPointRecipe

# raylib stores quaternions as Vector4 (x, y, z, w); stubs have no Quaternion type
type Quaternion = Vector4


@dataclass(frozen=True)
class MeshId:
    """stable handle into the mesh table"""

    name: str


class NodeRole(IntEnum):
    none = 0
    path_point = 1
    path_handle_in = 2
    path_handle_out = 3


@dataclass(frozen=True)
class Node:
    """a scene node; mesh_id none means transform-only group

    name is a display label shown in the outliner; empty means derive one.
    role tags path control points / handles for splines and vector shapes.
    generator, heightfield, vector_*, and road_sections_* recipes are mutually exclusive
    path_point recipe is only valid with role=path_point
    """

    id: str
    parent_id: str | None
    transform: Transform
    mesh_id: MeshId | None
    generator: Generator | None = None
    heightfield: Heightfield | None = None
    vector_layer: VectorLayerRecipe | None = None
    vector_shape: VectorShapeRecipe | None = None
    road_sections_layer: RoadSectionsLayerRecipe | None = None
    road_shp: RoadShpRecipe | None = None
    road_usage_group: RoadUsageGroupRecipe | None = None
    path_point: PathPointRecipe | None = None
    name: str = ""
    role: NodeRole = NodeRole.none


def transform_at(translation: Vector3) -> Transform:
    """return transform with identity rotation and unit scale"""
    return Transform(translation, quaternion_identity(), Vector3(1.0, 1.0, 1.0))


builtin_cube: MeshId = MeshId("cube")
builtin_cylinder: MeshId = MeshId("cylinder")
builtin_cone: MeshId = MeshId("cone")
builtin_torus: MeshId = MeshId("torus")
builtin_sphere: MeshId = MeshId("sphere")
builtin_mesh_ids: frozenset[MeshId] = frozenset(
    {
        builtin_cube,
        builtin_cylinder,
        builtin_cone,
        builtin_torus,
        builtin_sphere,
    }
)
