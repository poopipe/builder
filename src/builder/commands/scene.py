"""scene mutation commands"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from pyray import Transform, Vector3, vector3_add

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.generators.generator_types import (
    Generator,
    GeneratorSpec,
    MeshPattern,
    ParamMap,
)
from builder.generators.registry import default_params, get_spec
from builder.meshes.mesh_catalog import MeshAsset
from builder.scene.clone import clone_subtrees
from builder.scene.ids import new_node_id
from builder.scene.selection import subtree_ids
from builder.scene.scene_types import MeshId, Node, transform_at
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
    params: ParamMap = default_params(kind)
    mesh_id: MeshId = context.application.active_mesh_id
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    generator: Generator = Generator(
        kind=kind,
        meshes=MeshPattern(mesh_ids=(mesh_id,)),
        params=params,
    )
    locals_: list[Transform] = spec.build_transforms(params)
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.0, 0.0)),
        locals_,
        mesh_id,
        generator,
        name=spec.label,
    )
    context.scene.add_nodes(nodes)
    context.scene.set_selection([nodes[0].id])
    context.ui.status = f"Placed {spec.label} ({len(nodes) - 1} × {asset.label})"


def place_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "grid")


def place_radial_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "radial_grid")


def delete_selection(context: CommandContext, _: None) -> None:
    """remove selected groups and their descendants"""
    selected: list[str] = list(context.scene.nodes.selected_ids)
    if not selected:
        context.ui.status = "Nothing selected"
        return
    to_remove: list[str] = subtree_ids(context.scene.nodes.nodes, selected)
    context.scene.nodes.remove_nodes(to_remove)
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
    context.scene.add_nodes(clones)
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
cmd_place_radial_grid: Command[None] = Command("Radial grid", place_radial_grid)
cmd_delete: Command[None] = Command("Delete", delete_selection)
cmd_duplicate: Command[None] = Command("Duplicate", duplicate_selection)
cmd_group: Command[None] = Command("Group", group_selection)
cmd_parent: Command[None] = Command("Parent", parent_selection)
cmd_unparent: Command[None] = Command("Unparent", unparent_selection)
