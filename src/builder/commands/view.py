"""View / UI shell commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command


def toggle_grid(context: CommandContext, _: None) -> None:
    context.scene.toggle_grid()
    context.ui.status = f"Grid {'on' if context.scene.show_grid else 'off'}"


def focus_camera(context: CommandContext, _: None) -> None:
    context.scene.focus_origin()
    context.ui.status = "Camera focused on origin"


def toggle_side_panel(context: CommandContext, _: None) -> None:
    context.ui.toggle_side_panel()
    state = "shown" if context.ui.side_panel_open else "hidden"
    context.ui.status = f"Side panel {state}"


cmd_toggle_grid: Command[None] = Command("Grid", toggle_grid)
cmd_focus_camera: Command[None] = Command("Focus camera", focus_camera)
cmd_toggle_side_panel: Command[None] = Command("Panel", toggle_side_panel)
