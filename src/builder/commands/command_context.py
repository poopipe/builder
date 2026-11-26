"""protocols for what commands may interact with"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from builder.io.mesh_import import ImportedMesh, MeshImporter
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.road_sections.part_catalog import RoadPartsCatalog
from builder.heightfield.road_sections.road_sections_types import RoadEndpointRef
from builder.meshes.mesh_catalog import MeshCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.session_journal import SessionJournal
from builder.ui.dock import DockPanel
from builder.ui.file_browser import FileBrowserPurpose, FileBrowserState
from builder.view.gizmo import GizmoState
from builder.view.prepare_mesh import PreparedMesh
from builder.view.viewport import OrbitState
from pyray import Shader


class SceneContext(Protocol):
    """expose scene functions/state to commands"""

    show_grid: bool
    show_heightfield_wireframe: bool
    nodes: Scene
    gizmo: GizmoState
    orbit: OrbitState

    def toggle_grid(self) -> None:
        """show or hide the ground grid"""
        ...

    def toggle_heightfield_wireframe(self) -> None:
        """show or hide heightfield wireframe"""
        ...

    def focus_origin(self) -> None:
        """reset the camera on the origin"""
        ...

    def apply_orbit(self) -> None:
        """push orbit state into the live camera"""
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

    def register_mesh(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """insert or replace a prepared mesh in the mesh table"""
        ...

    def has_mesh(self, mesh_id: MeshId) -> bool:
        """true when the mesh table already has this id"""
        ...

    def unload_mesh(self, mesh_id: MeshId) -> None:
        """unload one prepared mesh if present; builtins are ignored"""
        ...

    def mesh_shader(self) -> Shader:
        """return the lighting shader used for mesh uploads"""
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
    height_layers_panel_open: bool
    sculpt_panel_open: bool
    outliner_open: bool
    left_dock_width: float
    right_dock_width: float
    left_dock_tab: DockPanel
    right_dock_tab: DockPanel
    meshes_panel_scroll: float
    file_browser: FileBrowserState | None
    browser_directories: dict[FileBrowserPurpose, Path]
    heightfield_layer_target: str | None
    heightfield_layer_replace_index: int | None
    road_sections_import_target: str | None
    road_polyline_usage_group_id: str | None
    road_polyline_ids: list[str]
    road_endpoints: list[RoadEndpointRef]

    def clear_road_polyline_selection(self) -> None:
        """clear compact road polyline / endpoint picks"""
        ...

    def toggle_side_panel(self) -> None:
        """show or hide the context side panel"""
        ...

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog panel"""
        ...

    def toggle_height_layers_panel(self) -> None:
        """show or hide the height layers panel"""
        ...

    def toggle_sculpt_panel(self) -> None:
        """show or hide the sculpt brush panel"""
        ...

    def toggle_outliner(self) -> None:
        """show or hide the scene outliner panel"""
        ...


class ApplicationContext(Protocol):
    """expose application functions/state to commands"""

    should_close: bool
    importer: MeshImporter
    mesh_catalog: MeshCatalog
    heightmap_catalog: HeightmapCatalog
    road_parts: RoadPartsCatalog
    active_mesh_id: MeshId
    scene_path: Path | None
    journal: SessionJournal


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
