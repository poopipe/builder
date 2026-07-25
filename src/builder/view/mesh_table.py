"""Table of prepared meshes keyed by MeshId."""

from __future__ import annotations

from dataclasses import dataclass, field

from builder.scene.scene_types import MeshId
from builder.view.prepare_mesh import PreparedMesh, release_prepared_mesh


@dataclass
class MeshTable:
    """Tracks prepared meshes for draw. No creation or shader logic."""

    entries: dict[MeshId, PreparedMesh] = field(default_factory=dict)

    def add(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """Store a prepared mesh. Raises if ``mesh_id`` is already present."""
        if mesh_id in self.entries:
            raise ValueError(f"mesh already registered: {mesh_id.name}")
        self.entries[mesh_id] = prepared

    def add_or_replace(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """ store a prepared mesh, unloading any previous entry with the same id """
        existing: PreparedMesh | None = self.entries.get(mesh_id)
        if existing is not None:
            release_prepared_mesh(existing)
        self.entries[mesh_id] = prepared

    def get(self, mesh_id: MeshId) -> PreparedMesh:
        """Return a prepared mesh. Raises ``KeyError`` if missing."""
        return self.entries[mesh_id]

    def has_mesh(self, mesh_id: MeshId) -> bool:
        """Return True if ``mesh_id`` is in the table."""
        return mesh_id in self.entries

    def unload(self) -> None:
        """Unload mesh GPU buffers.

        Materials are not unloaded: ``UnloadMaterial`` also frees
        ``material.shader``, and prepared meshes share an external shader.
        """
        prepared: PreparedMesh
        for prepared in self.entries.values():
            release_prepared_mesh(prepared)
        self.entries.clear()
