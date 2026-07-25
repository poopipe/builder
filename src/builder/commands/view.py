"""View / UI shell commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.view.gizmo_types import GizmoMode, GizmoSpace


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


def set_gizmo_mode(context: CommandContext, mode: GizmoMode) -> None:
    context.scene.gizmo.mode = mode
    context.ui.status = f"Gizmo mode: {mode.value}"


def set_gizmo_space(context: CommandContext, space: GizmoSpace) -> None:
    context.scene.gizmo.space = space
    context.ui.status = f"Gizmo space: {space.value}"


def toggle_gizmo_space(context: CommandContext, _: None) -> None:
    if context.scene.gizmo.space is GizmoSpace.world:
        set_gizmo_space(context, GizmoSpace.local)
    else:
        set_gizmo_space(context, GizmoSpace.world)


cmd_toggle_grid: Command[None] = Command("Grid", toggle_grid)
cmd_focus_camera: Command[None] = Command("Focus camera", focus_camera)
cmd_toggle_side_panel: Command[None] = Command("Panel", toggle_side_panel)
cmd_gizmo_translate: Command[GizmoMode] = Command("Move", set_gizmo_mode)
cmd_gizmo_rotate: Command[GizmoMode] = Command("Rotate", set_gizmo_mode)
cmd_gizmo_scale: Command[GizmoMode] = Command("Scale", set_gizmo_mode)
cmd_toggle_gizmo_space: Command[None] = Command("Local/World", toggle_gizmo_space)
