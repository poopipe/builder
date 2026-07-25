"""Scene mutation commands."""

from __future__ import annotations

from collections.abc import Sequence

from pyray import Transform, Vector3

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.generators.generator_types import Generator
from builder.generators.registry import default_params, get_spec
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


def place_generator_group(context: CommandContext, kind: str) -> None:
    """ place a parametric mesh group from a registry kind """
    spec = get_spec(kind)
    params = default_params(kind)
    generator: Generator = Generator(
        kind=kind,
        mesh_id=BUILTIN_CUBE,
        params=params,
    )
    locals_: list[Transform] = spec.build_transforms(params)
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.75, 0.0)),
        locals_,
        generator.mesh_id,
        generator,
    )
    context.scene.add_nodes(nodes)
    context.scene.set_selection([nodes[0].id])
    context.ui.status = f"Placed {spec.label} ({len(nodes) - 1} cubes)"


def place_cube_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "horizontal_grid")


def place_radial_grid(context: CommandContext, _: None) -> None:
    place_generator_group(context, "radial_grid")


cmd_place_cube_grid: Command[None] = Command("Cube grid", place_cube_grid)
cmd_place_radial_grid: Command[None] = Command("Radial grid", place_radial_grid)
