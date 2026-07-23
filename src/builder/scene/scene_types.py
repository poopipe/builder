"""Scene value types: mesh ids and nodes."""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Transform, Vector3, quaternion_identity


@dataclass(frozen=True)
class MeshId:
    """Stable handle into the mesh table."""

    name: str


@dataclass(frozen=True)
class Node:
    """A placed instance of a mesh in the scene."""

    id: str
    mesh_id: MeshId
    transform: Transform


def transform_at(translation: Vector3) -> Transform:
    """ return transform with identity rotation and unit scale """
    return Transform(translation, quaternion_identity(), Vector3(1.0, 1.0, 1.0))


BUILTIN_CUBE: MeshId = MeshId("cube")
