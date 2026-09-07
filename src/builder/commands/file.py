"""file / session commands"""

from __future__ import annotations

from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.heightfield.heightfield_types import HeightmapLayer
from builder.heightfield.heightmap_catalog import (
    HeightmapAsset,
    ensure_heightmap_buffer,
    heightmap_id_for_path,
    register_heightmap_asset,
)
from builder.heightfield.regenerate import set_heightfield
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.io.scene_format import scene_file_suffix
from builder.io.scene_io import load_scene_from_path, save_scene_to_path
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind, register_mesh_asset
from builder.scene.scene_types import MeshId, Node
from builder.ui.file_browser import (
    FileBrowserPurpose,
    FileBrowserState,
    make_import_browser,
    make_open_browser,
    make_save_browser,
    starting_directory,
)

mesh_file_suffix: str = ".fbx"
heightmap_file_suffix: str = ".png"


def remembered_browser_directory(
    context: CommandContext,
    purpose: FileBrowserPurpose,
) -> Path:
    """last folder for this purpose, else the scene project root / cwd"""
    last: Path | None = context.application.browser_directories.get(purpose)
    if last is not None and last.is_dir():
        return last.resolve()
    return starting_directory(context.application.scene_path)


def remember_browser_directory(
    context: CommandContext,
    browser: FileBrowserState,
) -> None:
    """store the browser's current folder for this purpose"""
    context.application.browser_directories[browser.purpose] = (
        browser.directory.resolve()
    )


def quit_app(context: CommandContext, _: None) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext, _: None) -> None:
    """open the in-app browser to load a .scene"""
    context.ui.file_browser = make_open_browser(
        remembered_browser_directory(context, FileBrowserPurpose.open_scene),
        scene_file_suffix,
        purpose=FileBrowserPurpose.open_scene,
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
    initial_name: str = (
        current.name if current is not None else f"untitled{scene_file_suffix}"
    )
    context.ui.file_browser = make_save_browser(
        remembered_browser_directory(context, FileBrowserPurpose.save_scene),
        scene_file_suffix,
        initial_name,
        purpose=FileBrowserPurpose.save_scene,
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
    """open the in-app browser to import an .fbx into the mesh catalog"""
    context.ui.file_browser = make_import_browser(
        remembered_browser_directory(context, FileBrowserPurpose.import_mesh),
        mesh_file_suffix,
    )
    context.ui.status = "Import mesh…"


def import_mesh_from_path(context: CommandContext, path: str | Path) -> None:
    """import an fbx into the mesh catalog and make it active"""
    mesh_path: Path = Path(path).resolve()
    imported: ImportedMesh = context.application.importer.import_path(mesh_path)
    mesh_id: MeshId = MeshId(mesh_id_for_import(imported))
    replacing: bool = mesh_id in context.application.mesh_catalog.entries
    registered_id: MeshId = context.scene.register_imported_mesh(imported)
    asset: MeshAsset = MeshAsset(
        mesh_id=registered_id,
        label=imported.name,
        kind=MeshAssetKind.fbx,
        source_path=imported.source_path,
    )
    register_mesh_asset(context.application.mesh_catalog, asset)
    context.application.active_mesh_id = registered_id
    context.application.browser_directories[FileBrowserPurpose.import_mesh] = (
        mesh_path.parent
    )
    if replacing:
        context.ui.status = (
            f"Replaced '{imported.name}' "
            f"({imported.triangle_count} tris)"
        )
        return
    context.ui.status = (
        f"Imported '{imported.name}' "
        f"({imported.triangle_count} tris)"
    )


def apply_import_mesh_path(context: CommandContext, path: Path) -> None:
    """import a chosen .fbx path from the file browser"""
    try:
        import_mesh_from_path(context, path)
    except (OSError, RuntimeError, ValueError) as exc:
        context.ui.status = f"Import failed: {exc}"


def apply_import_heightmap_path(context: CommandContext, path: Path) -> None:
    """attach a chosen heightmap image to the pending heightfield group"""
    from dataclasses import replace

    group_id: str | None = context.ui.heightfield_layer_target
    context.ui.heightfield_layer_target = None
    if group_id is None:
        context.ui.status = "No heightfield selected for heightmap"
        return
    group: Node | None = context.scene.nodes.nodes.get(group_id)
    if group is None or group.heightfield is None:
        context.ui.status = "Heightfield no longer available"
        return
    image_path: Path = path.resolve()
    heightmap_id = heightmap_id_for_path(image_path)
    # keep ids unique when the stem collides
    if heightmap_id in context.application.heightmap_catalog.entries:
        heightmap_id = heightmap_id_for_path(
            image_path.with_name(f"{image_path.stem}_{len(context.application.heightmap_catalog.entries)}{image_path.suffix}")
        )
    asset: HeightmapAsset = HeightmapAsset(
        heightmap_id=heightmap_id,
        label=image_path.stem,
        source_path=str(image_path),
    )
    try:
        register_heightmap_asset(context.application.heightmap_catalog, asset)
        ensure_heightmap_buffer(context.application.heightmap_catalog, heightmap_id)
    except (OSError, RuntimeError, ValueError) as exc:
        context.ui.status = f"Heightmap failed: {exc}"
        return
    layers: tuple[HeightmapLayer, ...] = group.heightfield.layers + (
        HeightmapLayer(heightmap_id=heightmap_id, amplitude=1.0, offset=0.0),
    )
    set_heightfield(
        context.scene,
        context.application.heightmap_catalog,
        context.scene.mesh_shader(),
        group_id,
        replace(group.heightfield, layers=layers),
    )
    context.application.browser_directories[FileBrowserPurpose.import_heightmap] = (
        image_path.parent
    )
    context.ui.status = f"Added heightmap '{asset.label}'"


cmd_quit: Command[None] = Command("Quit", quit_app)
cmd_open: Command[None] = Command("Open", open_scene)
cmd_save: Command[None] = Command("Save", save_scene)
cmd_save_as: Command[None] = Command("Save As", save_scene_as)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
