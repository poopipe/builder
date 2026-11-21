"""rebuild heightfield recipe residency (gpu draw owns meshes; no tile markers)"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from builder.heightfield.heightfield_types import (
    Heightfield,
    catalog_ids_for_heightfield,
    heightfield_layout,
    normalize_heightfield,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapCatalog,
    HeightmapId,
    ensure_heightmap_buffer,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.view.prepare_mesh import PreparedMesh


class HeightfieldSceneHost(Protocol):
    """scene + mesh table surface used when rebuilding heightfield tiles"""

    nodes: Scene

    def register_mesh(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """insert or replace a prepared mesh"""
        ...

    def unload_mesh(self, mesh_id: MeshId) -> None:
        """unload one prepared mesh if present"""
        ...


def tile_mesh_id(group_id: str, tile_x: int, tile_z: int) -> MeshId:
    """deterministic mesh id for one heightfield tile (lod/streaming hook)"""
    return MeshId(f"hf_{group_id}_{tile_x}_{tile_z}")


def tile_node_id(group_id: str, tile_x: int, tile_z: int) -> str:
    """deterministic node id for one heightfield tile marker (lod/streaming hook)"""
    return f"hf_tile_{group_id}_{tile_x}_{tile_z}"


def is_heightfield_tile_child(group_id: str, node: Node) -> bool:
    """true if node is a regenerated tile under the heightfield group"""
    if node.parent_id != group_id:
        return False
    if node.id.startswith(f"hf_tile_{group_id}_"):
        return True
    if node.mesh_id is not None and node.mesh_id.name.startswith(f"hf_{group_id}_"):
        return True
    return False


def heightfield_tile_child_ids(nodes: dict[str, Node], group_id: str) -> list[str]:
    """return ids of regenerated tile children of a heightfield group"""
    return [
        node.id
        for node in nodes.values()
        if is_heightfield_tile_child(group_id, node)
    ]


def selected_heightfield_group(scene: Scene) -> Node | None:
    """return the sole selected group if it owns a heightfield"""
    if len(scene.selected_ids) != 1:
        return None
    group_id: str = scene.selected_ids[0]
    node: Node | None = scene.nodes.get(group_id)
    if node is None or node.heightfield is None:
        return None
    return node


def clear_heightfield_tiles(host: HeightfieldSceneHost, group_id: str) -> None:
    """remove leftover tile children and unload this group's tile meshes only"""
    scene: Scene = host.nodes
    child_ids: list[str] = heightfield_tile_child_ids(scene.nodes, group_id)
    mesh_ids: list[MeshId] = []
    child_id: str
    for child_id in child_ids:
        child: Node = scene.nodes[child_id]
        if child.mesh_id is not None:
            mesh_ids.append(child.mesh_id)
    scene.remove_nodes(child_ids)
    prefix: str = f"hf_{group_id}_"
    mesh_id: MeshId
    for mesh_id in mesh_ids:
        if mesh_id.name.startswith(prefix):
            host.unload_mesh(mesh_id)


def preload_heightfield_buffers(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> None:
    """ensure cpu height buffers are resident for sampling and gpu upload"""
    heightmap_id: HeightmapId
    for heightmap_id in catalog_ids_for_heightfield(heightfield):
        ensure_heightmap_buffer(catalog, heightmap_id)


def regenerate_heightfield(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    group_id: str,
) -> None:
    """normalize the heightfield recipe, preload buffers, and drop leftover tiles

    gpu displace draws the surface; tile marker children are not created
    """
    scene: Scene = host.nodes
    group: Node = scene.nodes[group_id]
    heightfield: Heightfield | None = group.heightfield
    if heightfield is None:
        raise ValueError(f"group {group_id} has no heightfield")
    if group.generator is not None:
        raise ValueError(f"group {group_id} has both generator and heightfield")
    normalized: Heightfield = normalize_heightfield(heightfield)
    if normalized != heightfield:
        scene.add_nodes([replace(group, heightfield=normalized, generator=None)])
        heightfield = normalized
    preload_heightfield_buffers(catalog, heightfield)
    clear_heightfield_tiles(host, group_id)
    # validate size/spacing tile math; markers are not instantiated
    heightfield_layout(heightfield)


def set_heightfield(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
) -> None:
    """replace the heightfield recipe and refresh residency"""
    scene: Scene = host.nodes
    group: Node = scene.nodes[group_id]
    normalized: Heightfield = normalize_heightfield(heightfield)
    scene.add_nodes([replace(group, heightfield=normalized, generator=None)])
    regenerate_heightfield(host, catalog, group_id)
