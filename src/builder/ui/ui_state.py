"""mutable UI shell state"""

from __future__ import annotations

from dataclasses import dataclass, field

from builder.ui.file_browser import FileBrowserState
from builder.ui.text_field import FieldId, TextEdit


@dataclass
class UiState:
    """status text and panel visibility for the shell UI"""

    status: str = "Ready"
    side_panel_open: bool = True
    meshes_panel_open: bool = True
    outliner_open: bool = True
    meshes_panel_scroll: float = 0.0
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
    # group id waiting for a heightmap pick from the file browser
    heightfield_layer_target: str | None = None

    def toggle_side_panel(self) -> None:
        """show or hide the context side panel"""
        self.side_panel_open = not self.side_panel_open

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog panel"""
        self.meshes_panel_open = not self.meshes_panel_open

    def toggle_outliner(self) -> None:
        """show or hide the scene outliner panel"""
        self.outliner_open = not self.outliner_open

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
