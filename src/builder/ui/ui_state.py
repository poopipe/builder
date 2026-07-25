"""mutable UI shell state"""

from __future__ import annotations

from dataclasses import dataclass

from builder.ui.file_browser import FileBrowserState


@dataclass
class UiState:
    """status text and panel visibility for the shell UI"""

    status: str = "Ready"
    side_panel_open: bool = True
    meshes_panel_open: bool = True
    meshes_panel_scroll: float = 0.0
    inspector_group_id: str | None = None
    inspector_focus_key: str | None = None
    inspector_draft: str | None = None
    inspector_select_all: bool = False
    file_browser: FileBrowserState | None = None

    def toggle_side_panel(self) -> None:
        """show or hide the context side panel"""
        self.side_panel_open = not self.side_panel_open

    def toggle_meshes_panel(self) -> None:
        """show or hide the mesh catalog panel"""
        self.meshes_panel_open = not self.meshes_panel_open

    def clear_inspector_focus(self) -> None:
        """discard in-progress inspector text editing"""
        self.inspector_group_id = None
        self.inspector_focus_key = None
        self.inspector_draft = None
        self.inspector_select_all = False
