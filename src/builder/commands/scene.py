"""scene mutation commands"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pyray import Transform, Vector3, vector3_add

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.generators.generator_types import (
    Generator,
    GeneratorSpec,
    MeshPattern,
)
from builder.generators.registry import default_params, get_spec
from builder.generators.regenerate import regenerate_group
from builder.heightfield.heightfield_types import (
    HeightfieldLayout,
    default_heightfield,
    heightfield_layout,
)
from builder.heightfield.regenerate import regenerate_heightfield
from builder.meshes.mesh_catalog import MeshAsset
from builder.path.points import make_path_point
from builder.scene.clone import clone_subtrees
from builder.scene.ids import new_node_id
from builder.scene.selection import subtree_ids
from builder.scene.scene_types import (
    MeshId,
    Node,
    transform_at,
)
from builder.scene.transforms import (
    local_transform_under_parent,
    matrix_translation,
    world_matrix,
)


def build_mesh_group(
    group_transform: Transform,
    local_transforms: Sequence[Transform],
    mesh_id: MeshId,
    generator: Generator | None = None,
    name: str = "",
) -> list[Node]:
    """return group node plus meshed children with local transforms"""
    group_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=group_id,
            parent_id=None,
            transform=group_transform,
            mesh_id=None,
            generator=generator,
            name=name,
        )
    ]
    local: Transform
    for local in local_transforms:
        nodes.append(
            Node(
                id=new_node_id(),
                parent_id=group_id,
                transform=local,
                mesh_id=mesh_id,
                generator=None,
            )
        )
    return nodes


def default_spline_controls(group_id: str) -> list[Node]:
    """return a default open 3-point cubic path under group_id"""
    nodes: list[Node] = []
    specs: tuple[tuple[str, Vector3, Vector3 | None, Vector3 | None], ...] = (
        ("Point 0", Vector3(-4.0, 0.0, 0.0), None, Vector3(1.5, 0.0, 0.0)),
        (
            "Point 1",
            Vector3(0.0, 0.0, 0.0),
            Vector3(-1.5, 0.0, 0.0),
            Vector3(1.5, 0.0, 0.0),
        ),
        ("Point 2", Vector3(4.0, 0.0, 0.0), Vector3(-1.5, 0.0, 0.0), None),
    )
    name: str
    position: Vector3
    handle_in: Vector3 | None
    handle_out: Vector3 | None
    for name, position, handle_in, handle_out in specs:
        nodes.extend(
            make_path_point(
                group_id,
                name,
                position,
                handle_in=handle_in,
                handle_out=handle_out,
            )
        )
    return nodes


def place_mesh_group(context: CommandContext, mesh_id: MeshId, name: str) -> None:
    """place new mesh group"""
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.0, 0.0)),
        [transform_at(Vector3(0.0, 0.0, 0.0))],
        mesh_id,
        name=name,
    )
    context.scene.add_nodes(nodes)
    context.scene.set_selection([nodes[0].id])


def place_active_mesh(context: CommandContext, _: None) -> None:
    """place the active mesh asset"""
    mesh_id: MeshId = context.application.active_mesh_id
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    place_mesh_group(context, mesh_id, asset.label)
    context.ui.status = f"Placed {asset.label}"


def select_mesh(context: CommandContext, mesh_id: MeshId) -> None:
    """set the active mesh used by place and generator commands"""
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    context.application.active_mesh_id = mesh_id
    context.ui.status = f"Active mesh: {asset.label}"


def place_generator_group(context: CommandContext, kind: str) -> None:
    """place a parametric mesh group from a registry kind"""
    spec: GeneratorSpec = get_spec(kind)
    params: Any = default_params(kind)
    mesh_id: MeshId = context.application.active_mesh_id
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    pattern: MeshPattern = MeshPattern(mesh_ids=(mesh_id,))
    generator: Generator = Generator(
        kind=kind,
        meshes=pattern,
        params=params,
        point_meshes=pattern if spec.supports_point_meshes else None,
    )
    group_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=group_id,
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            generator=generator,
            name=spec.label,
        )
    ]
    if kind == "spline":
        nodes.extend(default_spline_controls(group_id))
    context.scene.add_nodes(nodes)
    regenerate_group(context.scene.nodes, group_id)
    meshed: int = len(
        [
            node
            for node in context.scene.nodes.nodes.values()
            if node.parent_id == group_id and node.mesh_id is not None
        ]
    )
    context.scene.set_selection([group_id])
    context.ui.status = f"Placed {spec.label} ({meshed} × {asset.label})"


def place_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "grid")


def place_radial(context: CommandContext, _: None) -> None:
    place_generator_group(context, "radial")


def place_ngon(context: CommandContext, _: None) -> None:
    place_generator_group(context, "ngon")


def place_spline(context: CommandContext, _: None) -> None:
    place_generator_group(context, "spline")


def place_heightfield(context: CommandContext, _: None) -> None:
    """place a tiled heightfield group and build its tile meshes"""
    group_id: str = new_node_id()
    heightfield = default_heightfield()
    context.scene.add_nodes(
        [
            Node(
                id=group_id,
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=heightfield,
                name="Heightfield",
            )
        ]
    )
    regenerate_heightfield(
        context.scene,
        context.application.heightmap_catalog,
        group_id,
    )
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    tiles: int = layout.tiles_x * layout.tiles_z
    context.scene.set_selection([group_id])
    context.ui.status = f"Placed Heightfield ({tiles} tiles)"


def delete_selection(context: CommandContext, _: None) -> None:
    """remove selected groups and their descendants"""
    selected: list[str] = list(context.scene.nodes.selected_ids)
    if not selected:
        context.ui.status = "Nothing selected"
        return
    to_remove: list[str] = subtree_ids(context.scene.nodes.nodes, selected)
    catalog_ids: frozenset[MeshId] = frozenset(
        context.application.mesh_catalog.entries.keys()
    )
    node_id: str
    for node_id in to_remove:
        node: Node | None = context.scene.nodes.nodes.get(node_id)
        if node is None or node.mesh_id is None:
            continue
        if node.mesh_id in catalog_ids:
            continue
        context.scene.unload_mesh(node.mesh_id)
    # capture vector layers touched before removal so remaining shapes re-rasterize
    from builder.heightfield.vector.edit import strip_vector_sources_from_heightfield
    from builder.heightfield.vector.rasterize import (
        find_heightfield_group_for_descendant,
        find_vector_layer_ancestor,
        regenerate_vectors_touching,
    )
    from builder.heightfield.regenerate import set_heightfield

    vector_touch: list[str] = []
    # heightfield id -> vector layer node ids being deleted with the selection
    hf_removed_layers: dict[str, set[str]] = {}
    removed_id: str
    for removed_id in to_remove:
        removed_node: Node | None = context.scene.nodes.nodes.get(removed_id)
        if removed_node is not None and removed_node.vector_layer is not None:
            hf_id: str | None = find_heightfield_group_for_descendant(
                context.scene.nodes.nodes, removed_id
            )
            if hf_id is not None and hf_id not in to_remove:
                hf_removed_layers.setdefault(hf_id, set()).add(removed_id)
        layer_id: str | None = find_vector_layer_ancestor(
            context.scene.nodes.nodes, removed_id
        )
        if layer_id is not None and layer_id not in to_remove:
            vector_touch.append(layer_id)
    context.scene.nodes.remove_nodes(to_remove)
    group_id: str
    layer_ids: set[str]
    for group_id, layer_ids in hf_removed_layers.items():
        group: Node | None = context.scene.nodes.nodes.get(group_id)
        if group is None or group.heightfield is None:
            continue
        stripped = strip_vector_sources_from_heightfield(
            group.heightfield, layer_ids
        )
        if stripped is not group.heightfield:
            set_heightfield(
                context.scene,
                context.application.heightmap_catalog,
                group_id,
                stripped,
            )
    if vector_touch:
        regenerate_vectors_touching(
            context.application.heightmap_catalog,
            context.scene.nodes.nodes,
            vector_touch,
        )
    context.scene.clear_selection()
    count: int = len(selected)
    label: str = "group" if count == 1 else "groups"
    context.ui.status = f"Deleted {count} {label}"


def duplicate_selection(context: CommandContext, _: None) -> None:
    """copy selected groups in place and select the copies"""
    selected: set[str] = set(context.scene.nodes.selected_ids)
    if not selected:
        context.ui.status = "Nothing selected"
        return
    # keep scene order so the copies are created in the same order as the originals
    roots: list[str] = [
        node_id for node_id in context.scene.nodes.nodes if node_id in selected
    ]
    clones: list[Node]
    new_roots: list[str]
    clones, new_roots = clone_subtrees(context.scene.nodes.nodes, roots)
    # drop cloned heightfield tile children; regenerate recreates markers
    kept: list[Node] = []
    clone: Node
    for clone in clones:
        if clone.parent_id in new_roots:
            root: Node | None = next(
                (node for node in clones if node.id == clone.parent_id), None
            )
            if root is not None and root.heightfield is not None:
                continue
        kept.append(clone)
    context.scene.add_nodes(kept)
    root_id: str
    for root_id in new_roots:
        root_node: Node | None = context.scene.nodes.nodes.get(root_id)
        if root_node is None or root_node.heightfield is None:
            continue
        regenerate_heightfield(
            context.scene,
            context.application.heightmap_catalog,
            root_id,
        )
    context.scene.set_selection(new_roots)
    count: int = len(new_roots)
    label: str = "group" if count == 1 else "groups"
    context.ui.status = f"Duplicated {count} {label}"


def is_ancestor(nodes: dict[str, Node], ancestor_id: str, node_id: str) -> bool:
    """true if ancestor_id sits above node_id in the parent chain"""
    current: Node | None = nodes.get(node_id)
    while current is not None and current.parent_id is not None:
        if current.parent_id == ancestor_id:
            return True
        current = nodes.get(current.parent_id)
    return False


def parent_selection(context: CommandContext, _: None) -> None:
    """reparent later-selected groups under the first-selected group"""
    selected: list[str] = list(context.scene.nodes.selected_ids)
    if len(selected) < 2:
        context.ui.status = "Select a parent, then the groups to parent under it"
        return
    nodes: dict[str, Node] = context.scene.nodes.nodes
    parent_id: str = selected[0]
    if parent_id not in nodes:
        context.ui.status = "Parent group no longer exists"
        return
    parented: int = 0
    child_id: str
    for child_id in selected[1:]:
        child: Node | None = nodes.get(child_id)
        if child is None or child_id == parent_id or child.parent_id == parent_id:
            continue
        # skip when the parent lives inside the child, which would loop the tree
        if is_ancestor(nodes, child_id, parent_id):
            continue
        local: Transform = local_transform_under_parent(nodes, child_id, parent_id)
        context.scene.nodes.set_node(
            replace(child, parent_id=parent_id, transform=local)
        )
        parented += 1
    # reparenting shifts descendant world matrices; force a full cache rebuild
    context.scene.nodes.mark_structure_changed()
    context.scene.set_selection([parent_id])
    parent_name: str = nodes[parent_id].name or "group"
    context.ui.status = f"Parented {parented} under {parent_name}"


def unparent_selection(context: CommandContext, _: None) -> None:
    """move selected groups back to the scene root, keeping world pose"""
    selected: list[str] = list(context.scene.nodes.selected_ids)
    if not selected:
        context.ui.status = "Nothing selected"
        return
    nodes: dict[str, Node] = context.scene.nodes.nodes
    unparented: int = 0
    node_id: str
    for node_id in selected:
        node: Node | None = nodes.get(node_id)
        if node is None or node.parent_id is None:
            continue
        local: Transform = local_transform_under_parent(nodes, node_id, None)
        context.scene.nodes.set_node(replace(node, parent_id=None, transform=local))
        unparented += 1
    context.scene.nodes.mark_structure_changed()
    context.ui.status = f"Unparented {unparented}"


def group_selection(context: CommandContext, _: None) -> None:
    """wrap selected groups in a new empty group at their centroid"""
    selected: list[str] = list(context.scene.nodes.selected_ids)
    if not selected:
        context.ui.status = "Nothing selected"
        return
    nodes: dict[str, Node] = context.scene.nodes.nodes
    total: Vector3 = Vector3(0.0, 0.0, 0.0)
    count: int = 0
    node_id: str
    for node_id in selected:
        if node_id in nodes:
            total = vector3_add(total, matrix_translation(world_matrix(nodes, node_id)))
            count += 1
    if count == 0:
        context.ui.status = "Nothing selected"
        return
    centroid: Vector3 = Vector3(total.x / count, total.y / count, total.z / count)
    group_id: str = new_node_id()
    context.scene.add_nodes(
        [
            Node(
                id=group_id,
                parent_id=None,
                transform=transform_at(centroid),
                mesh_id=None,
                generator=None,
                name="Group",
            )
        ]
    )
    for node_id in selected:
        node: Node | None = nodes.get(node_id)
        if node is None:
            continue
        local: Transform = local_transform_under_parent(nodes, node_id, group_id)
        context.scene.nodes.set_node(replace(node, parent_id=group_id, transform=local))
    context.scene.nodes.mark_structure_changed()
    context.scene.set_selection([group_id])
    context.ui.status = f"Grouped {count}"


cmd_place_mesh: Command[None] = Command("Place mesh", place_active_mesh)
cmd_select_mesh: Command[MeshId] = Command("Select mesh", select_mesh)
cmd_place_grid: Command[None] = Command("Grid", place_grid)
cmd_place_radial: Command[None] = Command("Radial", place_radial)
cmd_place_ngon: Command[None] = Command("N-gon", place_ngon)
cmd_place_spline: Command[None] = Command("Spline", place_spline)
cmd_place_heightfield: Command[None] = Command("Heightfield", place_heightfield)
cmd_delete: Command[None] = Command("Delete", delete_selection)
cmd_duplicate: Command[None] = Command("Duplicate", duplicate_selection)
cmd_group: Command[None] = Command("Group", group_selection)
cmd_parent: Command[None] = Command("Parent", parent_selection)
cmd_unparent: Command[None] = Command("Unparent", unparent_selection)
