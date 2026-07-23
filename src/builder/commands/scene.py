"""Scene mutation commands."""

from __future__ import annotations

from uuid import uuid4

from pyray import Transform, Vector3

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.distribution.grids import horizontal_grid_transforms
from builder.scene.scene_types import BUILTIN_CUBE, Node


def clear_selection(context: CommandContext, _: None) -> None:
    context.ui.status = "Selection cleared (placeholder)"


def place_example(context: CommandContext, _: None) -> None:
    context.ui.status = "Place: procedural placement not implemented yet"


def nudge_placeholder_up(context: CommandContext, steps: int) -> None:
    for _ in range(steps):
        context.scene.nudge_placeholder_up()
    context.ui.status = (
        "Nudged placeholder upward"
        if steps == 1
        else f"Nudged placeholder upward x{steps}"
    )


def place_cube_grid(context: CommandContext, _: None) -> None:
    transforms: list[Transform] = horizontal_grid_transforms(Vector3(0.0, 0.75, 0.0))
    batch_id: str = uuid4().hex[:8]
    nodes: list[Node] = []
    index: int
    transform: Transform
    for index, transform in enumerate(transforms):
        nodes.append(
            Node(
                id=f"cube_grid_{batch_id}_{index}",
                mesh_id=BUILTIN_CUBE,
                transform=transform,
            )
        )
    context.scene.add_nodes(nodes)
    context.ui.status = f"Placed {len(nodes)} cubes"


cmd_clear_selection: Command[None] = Command("Clear selection", clear_selection)
cmd_place_example: Command[None] = Command("Place example", place_example)
cmd_nudge_placeholder: Command[int] = Command("Nudge up", nudge_placeholder_up)
cmd_place_cube_grid: Command[None] = Command("Cube grid", place_cube_grid)
