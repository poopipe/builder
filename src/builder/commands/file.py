"""File / session commands."""

from __future__ import annotations

from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.io.mesh_import import ImportedMesh
from builder.scene.scene_types import MeshId


def quit_app(context: CommandContext, _: None) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext, _: None) -> None:
    context.ui.status = "Open: not implemented"


def about(context: CommandContext, _: None) -> None:
    context.ui.status = "Builder framework — pyray / raylib 6"


def import_mesh(context: CommandContext, _: None) -> None:
    context.ui.status = context.application.importer.status_message()


def import_mesh_from_path(context: CommandContext, path: str) -> None:
    """ load an fbx path, register it, and place one instance in the scene """
    imported: ImportedMesh = context.application.importer.import_path(path)
    mesh_id: MeshId = context.scene.register_and_place_imported_mesh(imported)
    context.ui.status = (
        f"Imported {Path(path).name} as '{imported.name}' "
        f"({imported.triangle_count} tris, id={mesh_id.name})"
    )


cmd_quit: Command[None] = Command("Quit", quit_app)
cmd_open: Command[None] = Command("Open", open_scene)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
