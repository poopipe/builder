"""session-level application state"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from builder.io.mesh_import import MeshImporter
from builder.meshes.mesh_catalog import MeshCatalog, default_mesh_catalog
from builder.scene.scene_types import builtin_cube, MeshId


@dataclass
class ApplicationState:
    """session services and selected mesh asset"""

    should_close: bool = False
    importer: MeshImporter = field(default_factory=MeshImporter)
    mesh_catalog: MeshCatalog = field(default_factory=default_mesh_catalog)
    active_mesh_id: MeshId = builtin_cube
    scene_path: Path | None = None
    # last folder opened per file-browser purpose (open_scene, save_scene, …)
    browser_directories: dict[str, Path] = field(default_factory=dict)
