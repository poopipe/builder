"""file / session commands"""

from __future__ import annotations

from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command
from builder.heightfield.asc_height_decode import AscImportHints, asc_import_hints
from builder.heightfield.height_export import export_composited_height
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
    normalize_heightfield,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    heightmap_id_for_path,
    import_heightmap_image,
)
from builder.heightfield.regenerate import set_heightfield
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.io.scene_format import scene_file_suffix
from builder.io.scene_io import load_scene_from_path, save_scene_to_path
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind, register_mesh_asset
from builder.scene.scene_types import MeshId, Node
from builder.session_journal import log_exception
from builder.ui.file_browser import (
    FileBrowserPurpose,
    FileBrowserState,
    make_import_browser,
    make_open_browser,
    make_save_browser,
    starting_directory,
)

mesh_file_suffix: str = ".fbx"
heightmap_file_suffixes: tuple[str, ...] = (".png", ".asc")


def apply_asc_base_import_hints(
    heightfield: Heightfield, hints: AscImportHints
) -> Heightfield:
    """apply ASC cellsize world size, map origin, and optional meta elevation"""
    from dataclasses import replace

    updated: Heightfield = replace(
        heightfield,
        size_x=replace(heightfield.size_x, value=hints.size_x),
        size_z=replace(heightfield.size_z, value=hints.size_z),
        origin_x=replace(heightfield.origin_x, value=hints.origin_x),
        origin_z=replace(heightfield.origin_z, value=hints.origin_z),
    )
    if hints.height_min is not None and hints.height_max is not None:
        updated = replace(
            updated,
            base_height_min=replace(
                updated.base_height_min, value=hints.height_min
            ),
            base_height_max=replace(
                updated.base_height_max, value=hints.height_max
            ),
        )
    return normalize_heightfield(updated)


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
        log_exception("save failed", exc)
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
    context.application.journal.record("open_scene", path=path)
    try:
        warnings: list[str] = load_scene_from_path(context, path)
    except (OSError, ValueError) as exc:
        log_exception("open scene failed", exc)
        context.ui.status = f"Open failed: {exc}"
        return
    if warnings:
        context.ui.status = f"Opened {path.name} ({len(warnings)} warning(s))"
        return
    context.ui.status = f"Opened {path.name}"


def apply_save_scene_path(context: CommandContext, path: Path) -> None:
    """save to a chosen .scene path from the file browser"""
    context.application.journal.record("save_scene", path=path)
    try:
        save_scene_to_path(context, path)
    except (OSError, ValueError) as exc:
        log_exception("save scene failed", exc)
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
    context.application.journal.record("import_mesh", path=path)
    try:
        import_mesh_from_path(context, path)
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception("import mesh failed", exc)
        context.ui.status = f"Import failed: {exc}"


def apply_import_heightmap_path(context: CommandContext, path: Path) -> None:
    """attach a chosen heightmap image to the pending heightfield group"""
    from dataclasses import replace

    group_id: str | None = context.ui.heightfield_layer_target
    replace_index: int | None = context.ui.heightfield_layer_replace_index
    context.ui.heightfield_layer_target = None
    context.ui.heightfield_layer_replace_index = None
    context.application.journal.record(
        "import_heightmap", path=path, group_id=group_id
    )
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
            image_path.with_name(
                f"{image_path.stem}_"
                f"{len(context.application.heightmap_catalog.entries)}"
                f"{image_path.suffix}"
            )
        )
    try:
        asset = import_heightmap_image(
            context.application.heightmap_catalog,
            image_path,
            heightmap_id,
            image_path.stem,
        )
        heightfield = group.heightfield
        status_verb: str
        if (
            replace_index is not None
            and 0 <= replace_index < len(heightfield.edit_layers)
        ):
            layers: list[HeightEditLayer] = list(heightfield.edit_layers)
            layers[replace_index] = replace(
                layers[replace_index],
                height=CatalogHeightSource(heightmap_id=heightmap_id),
                name=asset.label,
            )
            updated = replace(heightfield, edit_layers=tuple(layers))
            status_verb = "Replaced"
        elif heightfield.base is None:
            updated = replace(heightfield, base=heightmap_id)
            if image_path.suffix.lower() == ".asc":
                hints: AscImportHints = asc_import_hints(image_path)
                updated = apply_asc_base_import_hints(updated, hints)
            status_verb = "Set base"
        else:
            edit_layer: HeightEditLayer = HeightEditLayer(
                name=asset.label,
                height=CatalogHeightSource(heightmap_id=heightmap_id),
                blend=HeightBlendMode.add,
            )
            updated = replace(
                heightfield,
                edit_layers=heightfield.edit_layers + (edit_layer,),
            )
            status_verb = "Added"
        set_heightfield(
            context.scene,
            context.application.heightmap_catalog,
            group_id,
            updated,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception("import heightmap failed", exc)
        context.ui.status = f"Heightmap failed: {exc}"
        return
    context.application.browser_directories[FileBrowserPurpose.import_heightmap] = (
        image_path.parent
    )
    context.ui.status = (
        f"{status_verb} heightmap '{asset.label}' "
        f"({asset.width}x{asset.height} {asset.sample_type.name})"
    )


def apply_import_heightfield_albedo_path(
    context: CommandContext, path: Path
) -> None:
    """set the albedo image on the pending heightfield group"""
    from dataclasses import replace

    group_id: str | None = context.ui.heightfield_layer_target
    context.ui.heightfield_layer_target = None
    context.application.journal.record(
        "import_heightfield_albedo", path=path, group_id=group_id
    )
    if group_id is None:
        context.ui.status = "No heightfield selected for albedo"
        return
    group: Node | None = context.scene.nodes.nodes.get(group_id)
    if group is None or group.heightfield is None:
        context.ui.status = "Heightfield no longer available"
        return
    image_path: Path = path.resolve()
    try:
        set_heightfield(
            context.scene,
            context.application.heightmap_catalog,
            group_id,
            replace(group.heightfield, albedo_path=str(image_path)),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception("heightfield albedo failed", exc)
        context.ui.status = f"Albedo failed: {exc}"
        return
    context.application.browser_directories[
        FileBrowserPurpose.import_heightfield_albedo
    ] = image_path.parent
    context.ui.status = f"Albedo: {image_path.name}"


def apply_export_heightmap_path(context: CommandContext, path: Path) -> None:
    """write composited enabled-layer heights to .asc or .png"""
    group_id: str | None = context.ui.heightfield_layer_target
    context.ui.heightfield_layer_target = None
    context.ui.heightfield_layer_replace_index = None
    context.application.journal.record(
        "export_heightmap", path=path, group_id=group_id
    )
    if group_id is None:
        context.ui.status = "No heightfield selected for export"
        return
    group: Node | None = context.scene.nodes.nodes.get(group_id)
    if group is None or group.heightfield is None:
        context.ui.status = "Heightfield no longer available"
        return
    export_path: Path = path.resolve()
    try:
        buffer: HeightmapBuffer = export_composited_height(
            context.application.heightmap_catalog,
            group.heightfield,
            export_path,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception("export heightmap failed", exc)
        context.ui.status = f"Export failed: {exc}"
        return
    context.application.browser_directories[FileBrowserPurpose.export_heightmap] = (
        export_path.parent
    )
    context.ui.status = (
        f"Exported heightmap {export_path.name} "
        f"({buffer.width}x{buffer.height})"
    )


def apply_import_road_shp_path(context: CommandContext, path: Path) -> None:
    """import a shapefile under the pending road sections layer"""
    from builder.heightfield.road_sections.edit import import_road_shp
    from builder.heightfield.road_sections.road_sections_types import (
        road_usage_polyline_count,
    )

    layer_id: str | None = context.ui.road_sections_import_target
    context.ui.road_sections_import_target = None
    context.application.journal.record(
        "import_road_shp", path=path, layer_id=layer_id
    )
    if layer_id is None:
        context.ui.status = "No road sections layer selected for import"
        return
    layer: Node | None = context.scene.nodes.nodes.get(layer_id)
    if layer is None or layer.road_sections_layer is None:
        context.ui.status = "Road sections layer no longer available"
        return
    if layer.parent_id is None:
        context.ui.status = "Road sections layer has no heightfield parent"
        return
    parent: Node | None = context.scene.nodes.nodes.get(layer.parent_id)
    if parent is None or parent.heightfield is None:
        context.ui.status = "Heightfield parent no longer available"
        return
    shp_path: Path = path.resolve()
    try:
        new_nodes: list[Node] = import_road_shp(
            layer_id, shp_path, parent.heightfield
        )
    except (OSError, RuntimeError, ValueError) as exc:
        log_exception("import road shp failed", exc)
        context.ui.status = f"Import failed: {exc}"
        return
    context.scene.add_nodes(new_nodes)
    context.application.browser_directories[
        FileBrowserPurpose.import_road_shp
    ] = shp_path.parent
    polyline_count: int = 0
    node: Node
    for node in new_nodes:
        if node.road_usage_group is not None:
            polyline_count += road_usage_polyline_count(node.road_usage_group)
    context.ui.status = (
        f"Imported {shp_path.name} ({polyline_count} polylines)"
    )


cmd_quit: Command[None] = Command("Quit", quit_app)
cmd_open: Command[None] = Command("Open", open_scene)
cmd_save: Command[None] = Command("Save", save_scene)
cmd_save_as: Command[None] = Command("Save As", save_scene_as)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
