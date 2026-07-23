"""Mutable UI chrome state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UiState:
    """Status text and panel visibility for the shell UI."""

    status: str = "Ready"
    side_panel_open: bool = True

    def toggle_side_panel(self) -> None:
        """Show or hide the context side panel."""
        self.side_panel_open = not self.side_panel_open
