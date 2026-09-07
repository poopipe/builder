"""scene value types: mesh ids and nodes"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from pyray import Transform, Vector3, Vector4, quaternion_identity

if TYPE_CHECKING:
    from builder.generators.generator_types import Generator
    from builder.heightfield.heightfield_types import Heightfield

# raylib stores quaternions as Vector4 (x, y, z, w); stubs have no Quaternion type
type Quaternion = Vector4


@dataclass(frozen=True)
class MeshId:
    """stable handle into the mesh table"""

    name: str


class NodeRole(IntEnum):
    none = 0
    bezier_point = 1
    bezier_handle_in = 2
    bezier_handle_out = 3


@dataclass(frozen=True)
class Node:
    """a scene node; mesh_id none means transform-only group

    name is a display label shown in the outliner; empty means derive one.
    role tags control-point / handle nodes for spline generators.
    generator and heightfield are mutually exclusive recipes on a group
    """

    id: str
    parent_id: str | None
    transform: Transform
    mesh_id: MeshId | None
    generator: Generator | None = None
    heightfield: Heightfield | None = None
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
