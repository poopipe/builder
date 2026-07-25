"""File / session commands."""

from __future__ import annotations

from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.commands.scene import place_mesh_group
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.meshes.mesh_catalog import MeshAsset, register_mesh_asset
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
    """ import an fbx; replace GPU data if already loaded, else place one instance """
    imported: ImportedMesh = context.application.importer.import_path(path)
    mesh_id: MeshId = MeshId(mesh_id_for_import(imported))
    replacing: bool = mesh_id in context.application.mesh_catalog.entries
    registered_id: MeshId = context.scene.register_imported_mesh(imported)
    asset: MeshAsset = MeshAsset(
        mesh_id=registered_id,
        label=imported.name,
        kind="fbx",
        source_path=imported.source_path,
    )
    register_mesh_asset(context.application.mesh_catalog, asset)
    context.application.active_mesh_id = registered_id
    if replacing:
        context.ui.status = (
            f"Replaced '{imported.name}' "
            f"({imported.triangle_count} tris)"
        )
        return
    place_mesh_group(context, registered_id)
    context.ui.status = (
        f"Imported and selected {Path(path).name} as '{imported.name}' "
        f"({imported.triangle_count} tris)"
    )


cmd_quit: Command[None] = Command("Quit", quit_app)
cmd_open: Command[None] = Command("Open", open_scene)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
