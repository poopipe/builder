"""save and load .scene files against the live app/session"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from builder.commands.command_context import CommandContext
from builder.generators.generator_types import MeshPattern
from builder.generators.regenerate import regenerate_group
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.io.scene_format import (
    scene_format_version,
    SceneDocument,
    SceneUiState,
    dumps_document,
    loads_document,
    nodes_for_save,
)
from builder.meshes.mesh_catalog import (
    MeshAsset,
    MeshCatalog,
    default_mesh_catalog,
    register_mesh_asset,
)
from builder.scene.scene_types import builtin_cube, MeshId, Node


def project_root_for_scene(scene_path: Path) -> Path:
    """project root is the directory containing the .scene file"""
    return scene_path.resolve().parent


def relativize_path(path: Path, root: Path) -> str:
    """return path relative to root using forward slashes"""
    relative: Path = path.resolve().relative_to(root.resolve())
    return relative.as_posix()


def resolve_project_path(root: Path, relative: str) -> Path:
    """resolve a relative project path against root"""
    return (root / Path(relative)).resolve()


def build_document(context: CommandContext) -> SceneDocument:
    """snapshot the live session into a scene document"""
    return SceneDocument(
        version=scene_format_version,
        active_mesh_id=context.application.active_mesh_id,
        ui=SceneUiState(
            side_panel_open=context.ui.side_panel_open,
            meshes_panel_open=context.ui.meshes_panel_open,
            outliner_open=context.ui.outliner_open,
        ),
        meshes=tuple(context.application.mesh_catalog.entries.values()),
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
        if asset.kind != "fbx":
            continue
        if asset.source_path is None:
            raise ValueError(f"fbx asset '{asset.label}' has no source_path")
        sources[asset.mesh_id.name] = relativize_path(Path(asset.source_path), root)
    return sources


def save_scene_to_path(context: CommandContext, scene_path: Path) -> None:
    """write the current session to a .scene file"""
    path: Path = scene_path.resolve()
    root: Path = project_root_for_scene(path)
    document: SceneDocument = build_document(context)
    relative_sources: dict[str, str] = relative_sources_for_save(document, root)
    text: str = dumps_document(document, relative_sources)
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
    if mesh_id is node.mesh_id and remapped == pattern.mesh_ids:
        return node
    return replace(
        node,
        mesh_id=mesh_id,
        generator=replace(node.generator, meshes=replace(pattern, mesh_ids=remapped)),
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

    fallbacks: dict[str, MeshId] = {}
    asset: MeshAsset
    for asset in document.meshes:
        if asset.kind == "builtin":
            register_mesh_asset(
                catalog,
                MeshAsset(
                    mesh_id=asset.mesh_id,
                    label=asset.label,
                    kind="builtin",
                    source_path=None,
                ),
            )
            continue
        assert asset.source_path is not None
        absolute: Path = resolve_project_path(root, asset.source_path)
        try:
            imported: ImportedMesh = context.application.importer.import_path(absolute)
            registered_id: MeshId = context.scene.register_imported_mesh(imported)
            expected_id: MeshId = MeshId(mesh_id_for_import(imported))
            if registered_id != asset.mesh_id and expected_id != asset.mesh_id:
                warnings.append(
                    f"mesh id changed for '{asset.label}': "
                    f"file has {asset.mesh_id.name}, import produced {registered_id.name}"
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
                    kind="fbx",
                    source_path=str(absolute),
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(
                f"missing or invalid mesh '{asset.label}' "
                f"({asset.source_path}): {exc}; using cube"
            )
            fallbacks[asset.mesh_id.name] = builtin_cube

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
                warnings.append(f"failed to regenerate '{group.id}': {exc}")

    active: MeshId = remap_mesh_id(document.active_mesh_id, fallbacks) or builtin_cube
    if active not in catalog.entries:
        warnings.append(
            f"active mesh '{document.active_mesh_id.name}' unavailable; using cube"
        )
        active = builtin_cube
    context.application.active_mesh_id = active
    context.ui.side_panel_open = document.ui.side_panel_open
    context.ui.meshes_panel_open = document.ui.meshes_panel_open
    context.ui.outliner_open = document.ui.outliner_open
    context.application.scene_path = path
    return warnings
