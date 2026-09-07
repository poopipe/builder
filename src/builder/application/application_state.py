"""session-level application state"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.io.mesh_import import MeshImporter
from builder.meshes.mesh_catalog import MeshCatalog, default_mesh_catalog
from builder.scene.scene_types import builtin_cube, MeshId
from builder.ui.file_browser import FileBrowserPurpose


@dataclass
class ApplicationState:
    """session services and selected mesh asset"""

    should_close: bool = False
    importer: MeshImporter = field(default_factory=MeshImporter)
    mesh_catalog: MeshCatalog = field(default_factory=default_mesh_catalog)
    heightmap_catalog: HeightmapCatalog = field(default_factory=HeightmapCatalog)
    active_mesh_id: MeshId = builtin_cube
    scene_path: Path | None = None
    # last folder opened per file-browser purpose
    browser_directories: dict[FileBrowserPurpose, Path] = field(default_factory=dict)
