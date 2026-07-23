"""Scene mutation commands."""

from __future__ import annotations

from collections.abc import Sequence

from pyray import Transform, Vector3

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.distribution.grids import (
    horizontal_grid_transforms,
    radial_grid_transforms,
)
from builder.scene.ids import new_node_id
from builder.scene.scene_types import BUILTIN_CUBE, MeshId, Node, transform_at


def build_mesh_group(
    group_transform: Transform,
    local_transforms: Sequence[Transform],
    mesh_id: MeshId,
) -> list[Node]:
    """ return group node plus meshed children with local transforms """
    group_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=group_id,
            parent_id=None,
            transform=group_transform,
            mesh_id=None,
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
            )
        )
    return nodes


def place_cube_grid(context: CommandContext, _: None) -> None:
    """'_' arg is a placeholder for args passed to the function - not used in this case"""
    locals_: list[Transform] = horizontal_grid_transforms(Vector3(0.0, 0.0, 0.0))
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.75, 0.0)),
        locals_,
        BUILTIN_CUBE,
    )
    context.scene.add_nodes(nodes)
    context.ui.status = f"Placed cube grid ({len(nodes) - 1} cubes)"


def place_radial_grid(context: CommandContext, _: None) -> None:
    locals_: list[Transform] = radial_grid_transforms(Vector3(0.0, 0.0, 0.0), 5.0, 30.0)
    nodes: list[Node] = build_mesh_group(
        transform_at(Vector3(0.0, 0.75, 0.0)),
        locals_,
        BUILTIN_CUBE,
    )
    context.scene.add_nodes(nodes)
    context.ui.status = f"Placed radial grid ({len(nodes) - 1} cubes)"


cmd_place_cube_grid: Command[None] = Command("Cube grid", place_cube_grid)
cmd_place_radial_grid: Command[None] = Command("Radial grid", place_radial_grid)
