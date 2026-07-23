"""Built-in application commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.commands.registry import CommandRegistry


def quit_app(context: CommandContext) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext) -> None:
    context.ui.status = "Open: not implemented"


def about(context: CommandContext) -> None:
    context.ui.status = "Builder framework — pyray / raylib 6"


def import_mesh(context: CommandContext) -> None:
    context.ui.status = (
        "Import mesh: no file dialog yet. "
        + context.application.importer.status_message()
    )


def clear_selection(context: CommandContext) -> None:
    context.ui.status = "Selection cleared (placeholder)"


def place_example(context: CommandContext) -> None:
    context.ui.status = "Place: procedural placement not implemented yet"


def toggle_grid(context: CommandContext) -> None:
    assert context.scene is not None
    context.scene.toggle_grid()
    context.ui.status = f"Grid {'on' if context.scene.show_grid else 'off'}"


def focus_camera(context: CommandContext) -> None:
    assert context.scene is not None
    context.scene.focus_origin()
    context.ui.status = "Camera focused on origin"


def nudge_placeholder(context: CommandContext) -> None:
    """Example scene command."""
    assert context.scene is not None
    context.scene.nudge_placeholder_up()
    context.ui.status = "Nudged placeholder upward"


def toggle_side_panel(context: CommandContext) -> None:
    """Example UI command."""
    context.ui.toggle_side_panel()
    state = "shown" if context.ui.side_panel_open else "hidden"
    context.ui.status = f"Side panel {state}"


def register_builtin_commands(registry: CommandRegistry) -> None:
    """Register the default command set into the given registry."""
    builtins = [
        Command("quit", "Quit", quit_app),
        Command("open", "Open", open_scene),
        Command("about", "About", about),
        Command("toggle_side_panel", "Panel", toggle_side_panel),
        Command("import_mesh", "Import mesh", import_mesh),
        Command("toggle_grid", "Grid", toggle_grid),
        Command("focus_camera", "Focus camera", focus_camera),
        Command("nudge_placeholder", "Nudge up", nudge_placeholder),
        Command("clear_selection", "Clear selection", clear_selection),
        Command("place_example", "Place example", place_example),
    ]
    for command in builtins:
        registry.register(command)
