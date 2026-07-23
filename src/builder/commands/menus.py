"""Menu and panel command wiring."""

from __future__ import annotations

from builder.commands.commands_types import CommandEntry, CommandItem
from builder.commands.file import cmd_about, cmd_import_mesh, cmd_open, cmd_quit
from builder.commands.scene import (
    cmd_clear_selection,
    cmd_nudge_placeholder,
    cmd_place_cube_grid,
    cmd_place_example,
)
from builder.commands.view import (
    cmd_focus_camera,
    cmd_toggle_grid,
    cmd_toggle_side_panel,
)

MENU_COMMANDS: tuple[CommandItem, ...] = (
    CommandEntry(cmd_open, None),
    CommandEntry(cmd_quit, None),
    CommandEntry(cmd_about, None),
    CommandEntry(cmd_toggle_side_panel, None),
)
PANEL_COMMANDS: tuple[CommandItem, ...] = (
    CommandEntry(cmd_import_mesh, None),
    CommandEntry(cmd_toggle_grid, None),
    CommandEntry(cmd_focus_camera, None),
    CommandEntry(cmd_nudge_placeholder, 1),
    CommandEntry(cmd_place_cube_grid, None),
    CommandEntry(cmd_clear_selection, None),
    CommandEntry(cmd_place_example, None),
)


def all_commands() -> tuple[CommandItem, ...]:
    """Return every command entry exposed by the default menus."""
    return MENU_COMMANDS + PANEL_COMMANDS
