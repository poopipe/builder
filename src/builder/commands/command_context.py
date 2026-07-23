"""Protocols for what commands may interact with."""

from __future__ import annotations

from typing import Protocol

from builder.io.mesh_import import MeshImporter


class SceneContext(Protocol):
    """3D scene capabilities exposed to commands."""

    show_grid: bool

    def toggle_grid(self) -> None:
        """Show or hide the ground grid."""

    def focus_origin(self) -> None:
        """Reset the camera on the origin."""

    def nudge_placeholder_up(self) -> None:
        """Example scene mutation: raise the placeholder model."""


class UiContext(Protocol):
    """UI decoration and shell state exposed to commands."""

    status: str
    side_panel_open: bool

    def toggle_side_panel(self) -> None:
        """Show or hide the context side panel."""


class ApplicationContext(Protocol):
    """Session-level application capabilities exposed to commands."""

    should_close: bool
    importer: MeshImporter


class CommandContext(Protocol):
    """Composition root passed to every command."""

    @property
    def application(self) -> ApplicationContext:
        """Session / application state."""

    @property
    def scene(self) -> SceneContext | None:
        """Active 3D scene, if created."""

    @property
    def ui(self) -> UiContext:
        """UI shell state."""
