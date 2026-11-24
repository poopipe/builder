"""mutable UI shell state"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from builder.ui.dock import (
    DockPanel,
    DockResizeState,
    default_left_dock_width,
    default_right_dock_width,
)
from builder.ui.file_browser import FileBrowserPurpose, FileBrowserState
from builder.ui.text_field import FieldId, TextEdit


@dataclass(frozen=True)
class HeightfieldSizeDraft:
    """pending size_x/size_z edits waiting for apply"""

    group_id: str
    size_x: float
    size_z: float


@dataclass
class UiState:
    """status text and panel visibility for the shell UI"""

    status: str = "Ready"
    side_panel_open: bool = True
    meshes_panel_open: bool = True
    height_layers_panel_open: bool = True
    sculpt_panel_open: bool = True
    outliner_open: bool = True
    left_dock_width: float = field(default_factory=default_left_dock_width)
    right_dock_width: float = field(default_factory=default_right_dock_width)
    left_dock_tab: DockPanel = DockPanel.outliner
    right_dock_tab: DockPanel = DockPanel.meshes
    dock_resize: DockResizeState = field(default_factory=DockResizeState)
    meshes_panel_scroll: float = 0.0
    height_layers_panel_scroll: float = 0.0
    sculpt_panel_scroll: float = 0.0
    # selected edit layer index; none means base row
    height_layers_selected_index: int | None = None
    sculpt_brush_kind: str = "add"
    sculpt_radius: float = 32.0
    sculpt_intensity: float = 0.25
    outliner_scroll: float = 0.0
    # ids of collapsed outliner groups; not persisted with the scene
    outliner_collapsed: set[str] = field(default_factory=set)
    # titles of collapsed inspector param groups; Distribution starts open
    inspector_collapsed: set[str] = field(
        default_factory=lambda: {
            "Spacing",
            "Edge",
            "Point",
            "Orientation",
            "Edge meshes",
            "Point meshes",
            "Meshes",
        }
    )
    # last outliner row clicked and when, for double-click-to-rename detection
    outliner_click_id: str = ""
    outliner_click_time: float = 0.0
    focus: FieldId | None = None
    edit: TextEdit | None = None
    file_browser: FileBrowserState | None = None
    # last folder opened per file-browser purpose; persisted with the scene
    browser_directories: dict[FileBrowserPurpose, Path] = field(default_factory=dict)
    # group id waiting for a heightmap pick from the file browser
    heightfield_layer_target: str | None = None
    # when set with heightfield_layer_target, import replaces that edit layer texture
    heightfield_layer_replace_index: int | None = None
    # road sections layer id waiting for a shapefile pick
    road_sections_import_target: str | None = None
    # staged heightfield size until Apply size is pressed
    heightfield_size_draft: HeightfieldSizeDraft | None = None

    def toggle_side_panel(self) -> None:
        """show or hide the tools tab"""
        self.side_panel_open = not self.side_panel_open
        if self.side_panel_open:
            self.left_dock_tab = DockPanel.tools

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog tab"""
        self.meshes_panel_open = not self.meshes_panel_open
        if self.meshes_panel_open:
            self.right_dock_tab = DockPanel.meshes

    def toggle_height_layers_panel(self) -> None:
        """show or hide the height layers tab"""
        self.height_layers_panel_open = not self.height_layers_panel_open
        if self.height_layers_panel_open:
            self.right_dock_tab = DockPanel.height_layers

    def toggle_sculpt_panel(self) -> None:
        """show or hide the sculpt brush tab"""
        self.sculpt_panel_open = not self.sculpt_panel_open
        if self.sculpt_panel_open:
            self.right_dock_tab = DockPanel.sculpt

    def toggle_outliner(self) -> None:
        """show or hide the scene outliner tab"""
        self.outliner_open = not self.outliner_open
        if self.outliner_open:
            self.left_dock_tab = DockPanel.outliner

    def clear_focus(self) -> None:
        """discard in-progress text editing"""
        self.focus = None
        self.edit = None

    def focused_edit(self, panel: str, key: str, owner: str = "") -> TextEdit | None:
        """return the live edit when this exact field holds focus"""
        if self.focus is None or self.edit is None:
            return None
        if self.focus != FieldId(panel=panel, key=key, owner=owner):
            return None
        return self.edit
