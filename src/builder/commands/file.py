"""file / session commands"""

from __future__ import annotations

from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.commands.scene import place_mesh_group
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.io.scene_format import scene_file_suffix
from builder.io.scene_io import load_scene_from_path, save_scene_to_path
from builder.meshes.mesh_catalog import MeshAsset, register_mesh_asset
from builder.scene.scene_types import MeshId
from builder.ui.file_browser import (
    make_open_browser,
    make_save_browser,
    starting_directory,
)


def quit_app(context: CommandContext, _: None) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext, _: None) -> None:
    """open the in-app browser to load a .scene"""
    context.ui.file_browser = make_open_browser(
        starting_directory(context.application.scene_path),
        scene_file_suffix,
    )
    context.ui.status = "Open scene…"


def save_scene(context: CommandContext, _: None) -> None:
    """save to the current path, or open Save As when unset"""
    path: Path | None = context.application.scene_path
    if path is None:
        save_scene_as(context, None)
        return
    try:
        save_scene_to_path(context, path)
    except (OSError, ValueError) as exc:
        context.ui.status = f"Save failed: {exc}"
        return
    context.ui.status = f"Saved {path.name}"


def save_scene_as(context: CommandContext, _: None) -> None:
    """open the in-app browser to choose a .scene save path"""
    current: Path | None = context.application.scene_path
    initial_name: str = current.name if current is not None else f"untitled{scene_file_suffix}"
    context.ui.file_browser = make_save_browser(
        starting_directory(current),
        scene_file_suffix,
        initial_name,
    )
    context.ui.status = "Save scene as…"


def apply_open_scene_path(context: CommandContext, path: Path) -> None:
    """load a chosen .scene path from the file browser"""
    try:
        warnings: list[str] = load_scene_from_path(context, path)
    except (OSError, ValueError) as exc:
        context.ui.status = f"Open failed: {exc}"
        return
    if warnings:
        context.ui.status = f"Opened {path.name} ({len(warnings)} warning(s))"
        return
    context.ui.status = f"Opened {path.name}"


def apply_save_scene_path(context: CommandContext, path: Path) -> None:
    """save to a chosen .scene path from the file browser"""
    try:
        save_scene_to_path(context, path)
    except (OSError, ValueError) as exc:
        context.ui.status = f"Save failed: {exc}"
        return
    context.ui.status = f"Saved {path.name}"


def about(context: CommandContext, _: None) -> None:
    context.ui.status = "Builder framework — pyray / raylib 6"


def import_mesh(context: CommandContext, _: None) -> None:
    context.ui.status = context.application.importer.status_message()


def import_mesh_from_path(context: CommandContext, path: str) -> None:
    """import an fbx; replace GPU data if already loaded, else place one instance"""
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
cmd_save: Command[None] = Command("Save", save_scene)
cmd_save_as: Command[None] = Command("Save As", save_scene_as)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
