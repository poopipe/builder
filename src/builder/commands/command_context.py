"""protocols for what commands may interact with"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from builder.io.mesh_import import ImportedMesh, MeshImporter
from builder.meshes.mesh_catalog import MeshCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.ui.file_browser import FileBrowserState
from builder.view.gizmo import GizmoState


class SceneContext(Protocol):
    """expose scene functions/state to commands"""

    show_grid: bool
    nodes: Scene
    gizmo: GizmoState

    def toggle_grid(self) -> None:
        """show or hide the ground grid"""
        ...

    def focus_origin(self) -> None:
        """reset the camera on the origin"""
        ...

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """insert or replace scene nodes"""
        ...

    def clear_nodes(self) -> None:
        """remove every scene node"""
        ...

    def set_selection(self, group_ids: Sequence[str]) -> None:
        """replace the active group selection"""
        ...

    def clear_selection(self) -> None:
        """clear the active group selection"""
        ...

    def register_imported_mesh(self, imported: ImportedMesh) -> MeshId:
        """upload and register an imported mesh; return its mesh id"""
        ...

    def clear_non_builtin_meshes(self) -> None:
        """unload imported meshes; keep builtins"""
        ...

    def rebind_mesh_id(self, from_id: MeshId, to_id: MeshId) -> None:
        """rename a prepared mesh id"""
        ...


class UiContext(Protocol):
    """expose ui functions/state to commands"""

    status: str
    side_panel_open: bool
    meshes_panel_open: bool
    meshes_panel_scroll: float
    file_browser: FileBrowserState | None

    def toggle_side_panel(self) -> None:
        """show or hide the context side panel"""
        ...

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog panel"""
        ...


class ApplicationContext(Protocol):
    """expose application functions/state to commands"""

    should_close: bool
    importer: MeshImporter
    mesh_catalog: MeshCatalog
    active_mesh_id: MeshId
    scene_path: Path | None


class CommandContext(Protocol):
    """app, scene, and ui state handed to every command"""

    @property
    def application(self) -> ApplicationContext:
        """session and application state"""
        ...

    @property
    def scene(self) -> SceneContext:
        """active 3D scene"""
        ...

    @property
    def ui(self) -> UiContext:
        """UI shell state"""
        ...
