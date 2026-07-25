"""mutable UI shell state"""

from __future__ import annotations

from dataclasses import dataclass

from builder.ui.file_browser import FileBrowserState
from builder.ui.text_field import FieldId, TextEdit


@dataclass
class UiState:
    """status text and panel visibility for the shell UI"""

    status: str = "Ready"
    side_panel_open: bool = True
    meshes_panel_open: bool = True
    meshes_panel_scroll: float = 0.0
    focus: FieldId | None = None
    edit: TextEdit | None = None
    file_browser: FileBrowserState | None = None

    def toggle_side_panel(self) -> None:
        """show or hide the context side panel"""
        self.side_panel_open = not self.side_panel_open

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog panel"""
        self.meshes_panel_open = not self.meshes_panel_open

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
