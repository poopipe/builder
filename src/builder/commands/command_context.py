"""Protocols for what commands may interact with."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from builder.io.mesh_import import MeshImporter
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.view.gizmo import GizmoState


class SceneContext(Protocol):
    """expose scene functions/state to commands"""

    show_grid: bool
    nodes: Scene
    gizmo: GizmoState

    def toggle_grid(self) -> None:
        """Show or hide the ground grid."""
        ...

    def focus_origin(self) -> None:
        """Reset the camera on the origin."""
        ...

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """Insert or replace scene nodes."""
        ...

    def clear_nodes(self) -> None:
        """Remove every scene node."""
        ...

    def set_selection(self, group_ids: Sequence[str]) -> None:
        """Replace the active group selection."""
        ...

    def clear_selection(self) -> None:
        """Clear the active group selection."""
        ...


class UiContext(Protocol):
    """expose ui functions/state to commands"""

    status: str
    side_panel_open: bool

    def toggle_side_panel(self) -> None:
        """Show or hide the context side panel."""
        ...


class ApplicationContext(Protocol):
    """expose application functions/state to commands"""

    should_close: bool
    importer: MeshImporter


class CommandContext(Protocol):
    """app, scene, and ui state handed to every command"""

    @property
    def application(self) -> ApplicationContext:
        """Session and application state."""
        ...

    @property
    def scene(self) -> SceneContext:
        """Active 3D scene."""
        ...

    @property
    def ui(self) -> UiContext:
        """UI shell state."""
        ...
