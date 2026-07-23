"""Scene value types: mesh ids, transforms, nodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshId:
    """Stable handle into the mesh table."""

    name: str


@dataclass(frozen=True)
class Vec3:
    """Simple 3-vector for scene data (not a GPU type)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Transform:
    """TRS transform used by scene nodes."""

    position: Vec3 = Vec3()
    rotation: Vec3 = Vec3()  # euler radians (x, y, z)
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Node:
    """A placed instance of a mesh in the scene."""

    id: str
    mesh_id: MeshId
    transform: Transform


BUILTIN_CUBE: MeshId = MeshId("cube")
PLACEHOLDER_NODE_ID: str = "placeholder"
