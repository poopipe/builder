"""Built-in application commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import CommandItem, Command, CommandEntry


def quit_app(context: CommandContext, _: None) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext, _: None) -> None:
    context.ui.status = "Open: not implemented"


def about(context: CommandContext, _: None) -> None:
    context.ui.status = "Builder framework — pyray / raylib 6"


def import_mesh(context: CommandContext, _: None) -> None:
    context.ui.status = (
        "Import mesh: no file dialog yet. "
        + context.application.importer.status_message()
    )


def clear_selection(context: CommandContext, _: None) -> None:
    context.ui.status = "Selection cleared (placeholder)"


def place_example(context: CommandContext, _: None) -> None:
    context.ui.status = "Place: procedural placement not implemented yet"


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


def nudge_placeholder_up(context: CommandContext, steps: int) -> None:
    for _ in range(steps):
        context.scene.nudge_placeholder_up()
    context.ui.status = (
        "Nudged placeholder upward"
        if steps == 1
        else f"Nudged placeholder upward x{steps}"
    )


QUIT: Command[None] = Command("Quit", quit_app)
OPEN: Command[None] = Command("Open", open_scene)
ABOUT: Command[None] = Command("About", about)
TOGGLE_SIDE_PANEL: Command[None] = Command("Panel", toggle_side_panel)
IMPORT_MESH: Command[None] = Command("Import mesh", import_mesh)
TOGGLE_GRID: Command[None] = Command("Grid", toggle_grid)
FOCUS_CAMERA: Command[None] = Command("Focus camera", focus_camera)
NUDGE_PLACEHOLDER: Command[int] = Command("Nudge up", nudge_placeholder_up)
CLEAR_SELECTION: Command[None] = Command("Clear selection", clear_selection)
PLACE_EXAMPLE: Command[None] = Command("Place example", place_example)

MENU_COMMANDS: tuple[CommandItem, ...] = (
    CommandEntry(OPEN, None),
    CommandEntry(QUIT, None),
    CommandEntry(ABOUT, None),
    CommandEntry(TOGGLE_SIDE_PANEL, None),
)
PANEL_COMMANDS: tuple[CommandItem, ...] = (
    CommandEntry(IMPORT_MESH, None),
    CommandEntry(TOGGLE_GRID, None),
    CommandEntry(FOCUS_CAMERA, None),
    CommandEntry(NUDGE_PLACEHOLDER, 1),
    CommandEntry(CLEAR_SELECTION, None),
    CommandEntry(PLACE_EXAMPLE, None),
)


def builtin_commands() -> tuple[CommandItem, ...]:
    """Return the default command set."""
    return MENU_COMMANDS + PANEL_COMMANDS
