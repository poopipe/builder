"""save and load .scene files against the live app/session"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.generators.generator_types import MeshPattern
from builder.generators.regenerate import regenerate_group
from builder.heightfield.height_raw_cache import write_height_raw_cache
from builder.heightfield.heightmap_catalog import (
    HeightmapAsset,
    HeightmapBuffer,
    clear_heightmap_catalog,
    ensure_heightmap_buffer,
    register_heightmap_asset,
)
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.paint import decoded_f32_from_buffer
from builder.heightfield.regenerate import regenerate_heightfield
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.io.scene_format import (
    scene_format_version,
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    dumps_document,
    loads_document,
    nodes_for_save,
)
from pyray import Vector3
from builder.meshes.mesh_catalog import (
    MeshAsset,
    MeshAssetKind,
    MeshCatalog,
    default_mesh_catalog,
    register_mesh_asset,
)
from builder.scene.scene_types import builtin_cube, MeshId, Node
from builder.session_journal import log_exception
from builder.ui.zoom_bar import clamp_zoom_speed
from builder.view.orbit_move import clamp_move_speed, clamp_view_distance


def project_root_for_scene(scene_path: Path) -> Path:
    """project root is the directory containing the .scene file"""
    return scene_path.resolve().parent


def relativize_path(path: Path, root: Path) -> str:
    """return path relative to root using forward slashes; may include ..

    absolute path is kept when root and path are on different drives
    """
    resolved: Path = path.resolve()
    try:
        return resolved.relative_to(root.resolve(), walk_up=True).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_project_path(root: Path, relative: str) -> Path:
    """resolve a relative project path against root"""
    return (root / Path(relative)).resolve()


def build_document(context: CommandContext) -> SceneDocument:
    """snapshot the live session into a scene document"""
    target: Vector3 | None = context.scene.orbit.target
    if target is None:
        target = Vector3(0.0, 0.0, 0.0)
    return SceneDocument(
        version=scene_format_version,
        active_mesh_id=context.application.active_mesh_id,
        ui=SceneUiState(
            side_panel_open=context.ui.side_panel_open,
            meshes_panel_open=context.ui.meshes_panel_open,
            height_layers_panel_open=context.ui.height_layers_panel_open,
            sculpt_panel_open=context.ui.sculpt_panel_open,
            outliner_open=context.ui.outliner_open,
        ),
        camera=SceneCameraState(
            yaw=float(context.scene.orbit.yaw),
            pitch=float(context.scene.orbit.pitch),
            distance=float(context.scene.orbit.distance),
            target=Vector3(float(target.x), float(target.y), float(target.z)),
            zoom_speed=float(context.scene.orbit.zoom_speed),
            move_speed=float(context.scene.orbit.move_speed),
            view_distance=float(context.scene.orbit.view_distance),
        ),
        meshes=tuple(context.application.mesh_catalog.entries.values()),
        heightmaps=tuple(context.application.heightmap_catalog.entries.values()),
        nodes=tuple(nodes_for_save(context.scene.nodes.nodes)),
    )


def relative_sources_for_save(
    document: SceneDocument,
    root: Path,
) -> dict[str, str]:
    """map fbx mesh ids to paths relative to the scene project root"""
    sources: dict[str, str] = {}
    asset: MeshAsset
    for asset in document.meshes:
        match asset.kind:
            case MeshAssetKind.builtin:
                continue
            case MeshAssetKind.fbx:
                if asset.source_path is None:
                    raise ValueError(f"fbx asset '{asset.label}' has no source_path")
                sources[asset.mesh_id.name] = relativize_path(
                    Path(asset.source_path), root
                )
    return sources


def heightmap_sources_for_save(
    document: SceneDocument,
    root: Path,
) -> dict[str, str]:
    """map heightmap ids to source paths relative to the scene project root"""
    sources: dict[str, str] = {}
    asset: HeightmapAsset
    for asset in document.heightmaps:
        sources[asset.heightmap_id.name] = relativize_path(
            Path(asset.source_path), root
        )
    return sources


def heightmap_caches_for_save(
    document: SceneDocument,
    root: Path,
) -> dict[str, str]:
    """map heightmap ids to .hraw cache paths relative to the scene project root"""
    caches: dict[str, str] = {}
    asset: HeightmapAsset
    for asset in document.heightmaps:
        caches[asset.heightmap_id.name] = relativize_path(
            Path(asset.cache_path), root
        )
    return caches


def flush_paint_heightmaps(context: CommandContext) -> None:
    """write dirty paint (f32) buffers to their .hraw cache paths"""
    asset: HeightmapAsset
    for asset in context.application.heightmap_catalog.entries.values():
        if asset.sample_type is not HeightSampleType.f32:
            continue
        buffer: HeightmapBuffer | None = (
            context.application.heightmap_catalog.buffers.get(asset.heightmap_id)
        )
        if buffer is None:
            continue
        write_height_raw_cache(
            Path(asset.cache_path), decoded_f32_from_buffer(buffer)
        )


def save_scene_to_path(context: CommandContext, scene_path: Path) -> None:
    """write the current session to a .scene file"""
    path: Path = scene_path.resolve()
    root: Path = project_root_for_scene(path)
    flush_paint_heightmaps(context)
    document: SceneDocument = build_document(context)
    relative_sources: dict[str, str] = relative_sources_for_save(document, root)
    heightmap_sources: dict[str, str] = heightmap_sources_for_save(document, root)
    heightmap_caches: dict[str, str] = heightmap_caches_for_save(document, root)
    text: str = dumps_document(
        document, relative_sources, heightmap_sources, heightmap_caches
    )
    path.write_text(text, encoding="utf-8")
    context.application.scene_path = path


def remap_mesh_id(mesh_id: MeshId | None, fallbacks: dict[str, MeshId]) -> MeshId | None:
    """apply missing-asset fallbacks to a mesh id"""
    if mesh_id is None:
        return None
    return fallbacks.get(mesh_id.name, mesh_id)


def apply_mesh_fallbacks(node: Node, fallbacks: dict[str, MeshId]) -> Node:
    """rewrite node and generator mesh ids through the fallback map"""
    mesh_id: MeshId | None = remap_mesh_id(node.mesh_id, fallbacks)
    if node.generator is None:
        if mesh_id is node.mesh_id:
            return node
        return replace(node, mesh_id=mesh_id)
    pattern: MeshPattern = node.generator.meshes
    remapped: tuple[MeshId, ...] = tuple(
        remap_mesh_id(slot, fallbacks) or builtin_cube for slot in pattern.mesh_ids
    )
    point_pattern: MeshPattern | None = node.generator.point_meshes
    remapped_points: MeshPattern | None = point_pattern
    if point_pattern is not None:
        point_ids: tuple[MeshId, ...] = tuple(
            remap_mesh_id(slot, fallbacks) or builtin_cube
            for slot in point_pattern.mesh_ids
        )
        remapped_points = replace(point_pattern, mesh_ids=point_ids)
    if (
        mesh_id is node.mesh_id
        and remapped == pattern.mesh_ids
        and remapped_points == point_pattern
    ):
        return node
    return replace(
        node,
        mesh_id=mesh_id,
        generator=replace(
            node.generator,
            meshes=replace(pattern, mesh_ids=remapped),
            point_meshes=remapped_points,
        ),
    )


def load_scene_from_path(context: CommandContext, scene_path: Path) -> list[str]:
    """replace the live session from a .scene file; return warnings"""
    path: Path = scene_path.resolve()
    text: str = path.read_text(encoding="utf-8")
    document: SceneDocument
    warnings: list[str]
    document, warnings = loads_document(text)
    root: Path = project_root_for_scene(path)

    context.scene.clear_nodes()
    context.scene.clear_non_builtin_meshes()
    catalog: MeshCatalog = default_mesh_catalog()
    context.application.mesh_catalog = catalog
    context.application.active_mesh_id = builtin_cube
    clear_heightmap_catalog(context.application.heightmap_catalog)

    fallbacks: dict[str, MeshId] = {}
    asset: MeshAsset
    for asset in document.meshes:
        match asset.kind:
            case MeshAssetKind.builtin:
                register_mesh_asset(
                    catalog,
                    MeshAsset(
                        mesh_id=asset.mesh_id,
                        label=asset.label,
                        kind=MeshAssetKind.builtin,
                        source_path=None,
                    ),
                )
            case MeshAssetKind.fbx:
                assert asset.source_path is not None
                absolute: Path = resolve_project_path(root, asset.source_path)
                try:
                    imported: ImportedMesh = context.application.importer.import_path(
                        absolute
                    )
                    registered_id: MeshId = context.scene.register_imported_mesh(
                        imported
                    )
                    expected_id: MeshId = MeshId(mesh_id_for_import(imported))
                    if (
                        registered_id != asset.mesh_id
                        and expected_id != asset.mesh_id
                    ):
                        warnings.append(
                            f"mesh id changed for '{asset.label}': "
                            f"file has {asset.mesh_id.name}, "
                            f"import produced {registered_id.name}"
                        )
                    # keep the id from the scene file so node references stay valid
                    if registered_id != asset.mesh_id:
                        context.scene.rebind_mesh_id(registered_id, asset.mesh_id)
                        registered_id = asset.mesh_id
                    register_mesh_asset(
                        catalog,
                        MeshAsset(
                            mesh_id=registered_id,
                            label=asset.label,
                            kind=MeshAssetKind.fbx,
                            source_path=str(absolute),
                        ),
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    log_exception(
                        f"scene mesh load failed: {asset.label}", exc
                    )
                    warnings.append(
                        f"missing or invalid mesh '{asset.label}' "
                        f"({asset.source_path}): {exc}; using cube"
                    )
                    fallbacks[asset.mesh_id.name] = builtin_cube

    heightmap: HeightmapAsset
    for heightmap in document.heightmaps:
        absolute_hm: Path = resolve_project_path(root, heightmap.source_path)
        absolute_cache: Path = resolve_project_path(root, heightmap.cache_path)
        resolved: HeightmapAsset = HeightmapAsset(
            heightmap_id=heightmap.heightmap_id,
            label=heightmap.label,
            source_path=str(absolute_hm),
            cache_path=str(absolute_cache),
            sample_type=heightmap.sample_type,
            width=heightmap.width,
            height=heightmap.height,
        )
        register_heightmap_asset(context.application.heightmap_catalog, resolved)
        try:
            ensure_heightmap_buffer(
                context.application.heightmap_catalog, heightmap.heightmap_id
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log_exception(
                f"scene heightmap load failed: {heightmap.label}", exc
            )
            warnings.append(
                f"missing or invalid heightmap '{heightmap.label}' "
                f"({heightmap.source_path}): {exc}; treating as flat"
            )

    nodes: list[Node] = [
        apply_mesh_fallbacks(node, fallbacks) for node in document.nodes
    ]
    context.scene.add_nodes(nodes)

    group: Node
    for group in list(context.scene.nodes.nodes.values()):
        if group.generator is not None:
            try:
                regenerate_group(context.scene.nodes, group.id)
            except (KeyError, ValueError) as exc:
                log_exception(f"regenerate failed: {group.id}", exc)
                warnings.append(f"failed to regenerate '{group.id}': {exc}")
        if group.heightfield is not None:
            try:
                regenerate_heightfield(
                    context.scene,
                    context.application.heightmap_catalog,
                    group.id,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                log_exception(f"heightfield regenerate failed: {group.id}", exc)
                warnings.append(f"failed to regenerate heightfield '{group.id}': {exc}")

    active: MeshId = remap_mesh_id(document.active_mesh_id, fallbacks) or builtin_cube
    if active not in catalog.entries:
        warnings.append(
            f"active mesh '{document.active_mesh_id.name}' unavailable; using cube"
        )
        active = builtin_cube
    context.application.active_mesh_id = active
    context.ui.side_panel_open = document.ui.side_panel_open
    context.ui.meshes_panel_open = document.ui.meshes_panel_open
    context.ui.height_layers_panel_open = document.ui.height_layers_panel_open
    context.ui.sculpt_panel_open = document.ui.sculpt_panel_open
    context.ui.outliner_open = document.ui.outliner_open
    context.scene.orbit.yaw = document.camera.yaw
    context.scene.orbit.pitch = document.camera.pitch
    context.scene.orbit.distance = document.camera.distance
    context.scene.orbit.target = Vector3(
        document.camera.target.x,
        document.camera.target.y,
        document.camera.target.z,
    )
    context.scene.orbit.zoom_speed = clamp_zoom_speed(document.camera.zoom_speed)
    context.scene.orbit.move_speed = clamp_move_speed(document.camera.move_speed)
    context.scene.orbit.view_distance = clamp_view_distance(
        document.camera.view_distance
    )
    context.scene.apply_orbit()
    context.application.scene_path = path
    return warnings
