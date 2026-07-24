"""Scene value types: mesh ids and nodes."""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Transform, Vector3, Vector4, quaternion_identity

# raylib stores quaternions as Vector4 (x, y, z, w); stubs have no Quaternion type
type Quaternion = Vector4


@dataclass(frozen=True)
class MeshId:
    """Stable handle into the mesh table."""

    name: str


@dataclass(frozen=True)
class Node:
    """ a scene node; mesh_id none means transform-only group """

    id: str
    parent_id: str | None
    transform: Transform
    mesh_id: MeshId | None


def transform_at(translation: Vector3) -> Transform:
    """ return transform with identity rotation and unit scale """
    return Transform(translation, quaternion_identity(), Vector3(1.0, 1.0, 1.0))


BUILTIN_CUBE: MeshId = MeshId("cube")
