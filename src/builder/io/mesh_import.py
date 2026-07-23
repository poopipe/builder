"""Mesh import seam — stub until a cross-platform FBX loader is wired in.

Recommended later: ufbx via ``pyufbx`` or ``pufbx`` (no Autodesk SDK).
Assimp remains a fallback if ufbx wheels lag a given Python version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImportedMesh:
    """Minimal triangle mesh payload for future GPU upload."""

    name: str
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)


class MeshImporter:
    """Placeholder importer. Real FBX loading will plug in here."""

    supported_suffixes = (".fbx",)

    def can_import(self, path: str | Path) -> bool:
        """Return True if the path looks like a supported mesh file."""
        return Path(path).suffix.lower() in self.supported_suffixes

    def import_path(self, path: str | Path) -> ImportedMesh:
        """Attempt to import a mesh. Raises until a backend is installed."""
        path = Path(path)
        if not self.can_import(path):
            raise ValueError(f"unsupported mesh format: {path.suffix}")
        raise NotImplementedError(
            "FBX import is not wired yet. Prefer ufbx (pyufbx/pufbx) for "
            "cross-platform loading; avoid the Autodesk FBX SDK."
        )

    def status_message(self) -> str:
        """Human-readable backend status for the UI."""
        return "Mesh import: stub (install ufbx bindings later)"
