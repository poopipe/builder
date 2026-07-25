"""session catalog of meshes available to scene nodes"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from builder.scene.scene_types import builtin_cube, MeshId

type MeshAssetKind = Literal["builtin", "fbx"]


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
    cube: MeshAsset = MeshAsset(
        mesh_id=builtin_cube,
        label="Cube",
        kind="builtin",
    )
    return MeshCatalog(entries={cube.mesh_id: cube})


def register_mesh_asset(catalog: MeshCatalog, asset: MeshAsset) -> None:
    """add or replace mesh metadata by id"""
    catalog.entries[asset.mesh_id] = asset


def catalog_assets(catalog: MeshCatalog) -> list[MeshAsset]:
    """return catalog assets in insertion order"""
    return list(catalog.entries.values())
