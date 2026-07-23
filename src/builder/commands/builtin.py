"""Built-in application commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command


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
    context.scene.toggle_grid()
    context.ui.status = f"Grid {'on' if context.scene.show_grid else 'off'}"


def focus_camera(context: CommandContext) -> None:
    context.scene.focus_origin()
    context.ui.status = "Camera focused on origin"


def toggle_side_panel(context: CommandContext) -> None:
    context.ui.toggle_side_panel()
    state = "shown" if context.ui.side_panel_open else "hidden"
    context.ui.status = f"Side panel {state}"


def make_nudge_up_command(steps: int = 1) -> Command:
    """Build a scene nudge command; ``steps`` is closed over by ``run``."""

    def run(context: CommandContext) -> None:
        for _ in range(steps):
            context.scene.nudge_placeholder_up()
        context.ui.status = (
            "Nudged placeholder upward"
            if steps == 1
            else f"Nudged placeholder upward x{steps}"
        )

    label = "Nudge up" if steps == 1 else f"Nudge up x{steps}"
    return Command(label, run)


QUIT = Command("Quit", quit_app)
OPEN = Command("Open", open_scene)
ABOUT = Command("About", about)
TOGGLE_SIDE_PANEL = Command("Panel", toggle_side_panel)
IMPORT_MESH = Command("Import mesh", import_mesh)
TOGGLE_GRID = Command("Grid", toggle_grid)
FOCUS_CAMERA = Command("Focus camera", focus_camera)
NUDGE_PLACEHOLDER = make_nudge_up_command(1)
CLEAR_SELECTION = Command("Clear selection", clear_selection)
PLACE_EXAMPLE = Command("Place example", place_example)

MENU_COMMANDS: tuple[Command, ...] = (
    OPEN,
    QUIT,
    ABOUT,
    TOGGLE_SIDE_PANEL,
)
PANEL_COMMANDS: tuple[Command, ...] = (
    IMPORT_MESH,
    TOGGLE_GRID,
    FOCUS_CAMERA,
    NUDGE_PLACEHOLDER,
    CLEAR_SELECTION,
    PLACE_EXAMPLE,
)


def builtin_commands() -> tuple[Command, ...]:
    """Return the default command set."""
    return MENU_COMMANDS + PANEL_COMMANDS
