"""menu and panel command wiring"""

from __future__ import annotations

from builder.commands.commands_types import CommandEntry, CommandItem
from builder.commands.file import (
    cmd_about,
    cmd_import_mesh,
    cmd_open,
    cmd_quit,
    cmd_save,
    cmd_save_as,
)
from builder.commands.scene import (
    cmd_delete,
    cmd_place_grid,
    cmd_place_mesh,
    cmd_place_radial_grid,
)
from builder.commands.view import (
    cmd_focus_camera,
    cmd_gizmo_rotate,
    cmd_gizmo_scale,
    cmd_gizmo_translate,
    cmd_toggle_gizmo_space,
    cmd_toggle_grid,
    cmd_toggle_meshes_panel,
    cmd_toggle_side_panel,
)
from builder.view.gizmo_types import GizmoMode

menu_commands: tuple[CommandItem, ...] = (
    CommandEntry(cmd_open, None),
    CommandEntry(cmd_save, None),
    CommandEntry(cmd_save_as, None),
    CommandEntry(cmd_quit, None),
    CommandEntry(cmd_about, None),
    CommandEntry(cmd_toggle_side_panel, None),
    CommandEntry(cmd_toggle_meshes_panel, None),
    CommandEntry(cmd_gizmo_translate, GizmoMode.translate),
    CommandEntry(cmd_gizmo_rotate, GizmoMode.rotate),
    CommandEntry(cmd_gizmo_scale, GizmoMode.scale),
    CommandEntry(cmd_toggle_gizmo_space, None),
)
panel_commands: tuple[CommandItem, ...] = (
    CommandEntry(cmd_import_mesh, None),
    CommandEntry(cmd_place_mesh, None),
    CommandEntry(cmd_delete, None),
    CommandEntry(cmd_toggle_grid, None),
    CommandEntry(cmd_focus_camera, None),
    CommandEntry(cmd_place_grid, None),
    CommandEntry(cmd_place_radial_grid, None),
)


def all_commands() -> tuple[CommandItem, ...]:
    """return every command entry exposed by the default menus"""
    return menu_commands + panel_commands
