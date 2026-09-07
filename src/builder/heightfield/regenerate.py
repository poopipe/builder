"""rebuild heightfield tile children and gpu meshes"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from pyray import Shader, Vector3

from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.tile_mesh import field_height_range, upload_tile_mesh
from builder.scene.ids import new_node_id
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node, transform_at
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

    def mesh_shader(self) -> Shader:
        """lighting shader for mesh uploads"""
        ...


def tile_mesh_id(group_id: str, tile_x: int, tile_z: int) -> MeshId:
    """deterministic mesh id for one heightfield tile"""
    return MeshId(f"hf_{group_id}_{tile_x}_{tile_z}")


def heightfield_tile_child_ids(nodes: dict[str, Node], group_id: str) -> list[str]:
    """return ids of direct meshed children of a heightfield group"""
    return [
        node.id
        for node in nodes.values()
        if node.parent_id == group_id and node.mesh_id is not None
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
    """remove tile children and unload this group's tile meshes only"""
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


def regenerate_heightfield(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    shader: Shader,
    group_id: str,
) -> None:
    """replace tile children from the group's heightfield recipe"""
    scene: Scene = host.nodes
    group: Node = scene.nodes[group_id]
    heightfield: Heightfield | None = group.heightfield
    if heightfield is None:
        raise ValueError(f"group {group_id} has no heightfield")
    if group.generator is not None:
        raise ValueError(f"group {group_id} has both generator and heightfield")
    clear_heightfield_tiles(host, group_id)

    height_min: float
    height_max: float
    height_min, height_max = field_height_range(catalog, heightfield)

    tiles_x: int = heightfield.tiles_x.value
    tiles_z: int = heightfield.tiles_z.value
    tile_verts: int = heightfield.tile_verts.value
    step: int = tile_verts - 1
    res_x: int = tiles_x * step + 1
    res_z: int = tiles_z * step + 1
    cell_x: float = (
        heightfield.size_x.value / float(res_x - 1)
        if res_x > 1
        else heightfield.size_x.value
    )
    cell_z: float = (
        heightfield.size_z.value / float(res_z - 1)
        if res_z > 1
        else heightfield.size_z.value
    )
    # center the field on the group origin
    origin_x: float = -0.5 * heightfield.size_x.value
    origin_z: float = -0.5 * heightfield.size_z.value

    children: list[Node] = []
    tile_z: int
    tile_x: int
    for tile_z in range(tiles_z):
        for tile_x in range(tiles_x):
            mesh_id: MeshId = tile_mesh_id(group_id, tile_x, tile_z)
            prepared: PreparedMesh = upload_tile_mesh(
                catalog,
                heightfield,
                tile_x,
                tile_z,
                shader,
                height_min,
                height_max,
            )
            host.register_mesh(mesh_id, prepared)
            local_x: float = origin_x + float(tile_x * step) * cell_x
            local_z: float = origin_z + float(tile_z * step) * cell_z
            children.append(
                Node(
                    id=new_node_id(),
                    parent_id=group_id,
                    transform=transform_at(Vector3(local_x, 0.0, local_z)),
                    mesh_id=mesh_id,
                    name=f"Tile {tile_x},{tile_z}",
                )
            )
    scene.add_nodes(children)


def set_heightfield(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    shader: Shader,
    group_id: str,
    heightfield: Heightfield,
) -> None:
    """replace the heightfield recipe and regenerate tiles"""
    scene: Scene = host.nodes
    group: Node = scene.nodes[group_id]
    scene.add_nodes([replace(group, heightfield=heightfield, generator=None)])
    regenerate_heightfield(host, catalog, shader, group_id)
