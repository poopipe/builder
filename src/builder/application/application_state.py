"""Session-level application state."""

from __future__ import annotations

from dataclasses import dataclass, field

from builder.io.mesh_import import MeshImporter


@dataclass
class ApplicationState:
    """Quit flag and services owned by the running session."""

    should_close: bool = False
    importer: MeshImporter = field(default_factory=MeshImporter)
