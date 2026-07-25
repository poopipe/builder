"""Scene mutation commands."""

from __future__ import annotations

from collections.abc import Sequence

from pyray import Transform, Vector3

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.generators.generator_types import Generator, GeneratorSpec, ParamMap
from builder.generators.registry import default_params, get_spec
from builder.meshes.mesh_catalog import MeshAsset
from builder.scene.ids import new_node_id
from builder.scene.scene_types import BUILTIN_CUBE, MeshId, Node, transform_at


def build_mesh_group(
    group_transform: Transform,
    local_transforms: Sequence[Transform],
    mesh_id: MeshId,
    generator: Generator | None = None,
) -> list[Node]:
    """ return group node plus meshed children with local transforms """
    group_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=group_id,
            parent_id=None,
            transform=group_transform,
            mesh_id=None,
            generator=generator,
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


def place_mesh_group(context: CommandContext, mesh_id: MeshId) -> None:
    """ place one instance of a registered mesh under a selected group """
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.0, 0.0)),
        [transform_at(Vector3(0.0, 0.0, 0.0))],
        mesh_id,
    )
    context.scene.add_nodes(nodes)
    context.scene.set_selection([nodes[0].id])


def place_active_mesh(context: CommandContext, _: None) -> None:
    """ place the active mesh asset """
    mesh_id: MeshId = context.application.active_mesh_id
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    place_mesh_group(context, mesh_id)
    context.ui.status = f"Placed {asset.label}"


def select_mesh(context: CommandContext, mesh_id: MeshId) -> None:
    """ set the active mesh used by place and generator commands """
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    context.application.active_mesh_id = mesh_id
    context.ui.status = f"Active mesh: {asset.label}"


def place_generator_group(context: CommandContext, kind: str) -> None:
    """ place a parametric mesh group from a registry kind """
    spec: GeneratorSpec = get_spec(kind)
    params: ParamMap = default_params(kind)
    mesh_id: MeshId = context.application.active_mesh_id
    asset: MeshAsset | None = context.application.mesh_catalog.entries.get(mesh_id)
    if asset is None:
        context.ui.status = f"Mesh is not available: {mesh_id.name}"
        return
    generator: Generator = Generator(
        kind=kind,
        mesh_id=mesh_id,
        params=params,
    )
    locals_: list[Transform] = spec.build_transforms(params)
    group_y: float = 0.75 if mesh_id == BUILTIN_CUBE else 0.0
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, group_y, 0.0)),
        locals_,
        generator.mesh_id,
        generator,
    )
    context.scene.add_nodes(nodes)
    context.scene.set_selection([nodes[0].id])
    context.ui.status = (
        f"Placed {spec.label} ({len(nodes) - 1} × {asset.label})"
    )


def place_horizontal_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "horizontal_grid")


def place_radial_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "radial_grid")


cmd_place_mesh: Command[None] = Command("Place mesh", place_active_mesh)
cmd_select_mesh: Command[MeshId] = Command("Select mesh", select_mesh)
cmd_place_horizontal_grid: Command[None] = Command("Grid", place_horizontal_grid)
cmd_place_radial_grid: Command[None] = Command("Radial grid", place_radial_grid)
