"""session catalog of meshes available to scene nodes"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from builder.scene.scene_types import (
    MeshId,
    builtin_cone,
    builtin_cube,
    builtin_cylinder,
    builtin_sphere,
    builtin_torus,
)


class MeshAssetKind(IntEnum):
    builtin = 0
    fbx = 1


@dataclass(frozen=True)
class MeshAsset:
    """metadata for a mesh available in the session"""

    mesh_id: MeshId
    label: str
    kind: MeshAssetKind
    source_path: str | None = None


@dataclass
class MeshCatalog:
    """meshes available for placement and generators"""

    entries: dict[MeshId, MeshAsset] = field(default_factory=dict)


def default_mesh_catalog() -> MeshCatalog:
    """return a catalog containing built-in meshes"""
    assets: tuple[MeshAsset, ...] = (
        MeshAsset(mesh_id=builtin_cube, label="Cube", kind=MeshAssetKind.builtin),
        MeshAsset(
            mesh_id=builtin_cylinder, label="Cylinder", kind=MeshAssetKind.builtin
        ),
        MeshAsset(mesh_id=builtin_cone, label="Cone", kind=MeshAssetKind.builtin),
        MeshAsset(mesh_id=builtin_torus, label="Torus", kind=MeshAssetKind.builtin),
        MeshAsset(mesh_id=builtin_sphere, label="Sphere", kind=MeshAssetKind.builtin),
    )
    return MeshCatalog(entries={asset.mesh_id: asset for asset in assets})


def register_mesh_asset(catalog: MeshCatalog, asset: MeshAsset) -> None:
    """add or replace mesh metadata by id"""
    catalog.entries[asset.mesh_id] = asset
