"""Scene mutation commands."""

from __future__ import annotations

from pyray import Transform, Vector3

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.distribution.grids import (
    horizontal_grid_transforms,
    radial_grid_transforms,
)
from builder.scene.ids import new_node_id
from builder.scene.scene_types import BUILTIN_CUBE, Node


def place_cube_grid(context: CommandContext, _: None) -> None:
    """'_' arg is a placeholder for args passed to the function - not used in this case"""
    transforms: list[Transform] = horizontal_grid_transforms(Vector3(0.0, 0.75, 0.0))
    nodes: list[Node] = list()
    transform: Transform
    for transform in transforms:
        nodes.append(
            Node(
                id=new_node_id(),
                mesh_id=BUILTIN_CUBE,
                transform=transform,
            )
        )
    context.scene.add_nodes(nodes)
    context.ui.status = f"Placed {len(nodes)} cubes"


def place_radial_grid(context: CommandContext, _: None) -> None:
    transforms: list[Transform] = radial_grid_transforms(
        Vector3(0.0, 0.75, 0.0), 5.0, 30.0
    )
    nodes: list[Node] = list()
    transform: Transform
    for transform in transforms:
        nodes.append(Node(id=new_node_id(), mesh_id=BUILTIN_CUBE, transform=transform))
    context.scene.add_nodes(nodes)
    context.ui.status = f"Placed {len(nodes)} cubes"


cmd_place_cube_grid: Command[None] = Command("Cube grid", place_cube_grid)
cmd_place_radial_grid: Command[None] = Command("Radial grid", place_radial_grid)
