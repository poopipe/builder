"""FBX mesh import via the ufbx_bridge DLL (no numpy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from builder.io.ufbx_bridge import default_dll_path, load_fbx_via_bridge


@dataclass(frozen=True)
class ImportedMesh:
    """ triangulated mesh ready for GPU upload (non-indexed corner verts) """

    name: str
    source_path: str
    # flat triples: x,y,z per corner vertex
    positions: tuple[float, ...]
    normals: tuple[float, ...]
    # flat pairs: u,v per corner vertex (zeros if absent)
    texcoords: tuple[float, ...]

    @property
    def vertex_count(self) -> int:
        return len(self.positions) // 3

    @property
    def triangle_count(self) -> int:
        return self.vertex_count // 3


@dataclass
class MeshImporter:
    """ loads .fbx files into ImportedMesh payloads """

    supported_suffixes: tuple[str, ...] = (".fbx",)

    def has_valid_suffix(self, path: str | Path) -> bool:
        """ return true if the path has a supported mesh file suffix """
        return Path(path).suffix.lower() in self.supported_suffixes

    def status_message(self) -> str:
        """ human-readable backend status for the UI """
        if not default_dll_path().is_file():
            return (
                "Mesh import: build ufbx_bridge\\build\\ufbx_bridge.dll "
                "(run ufbx_bridge\\build.cmd), then drop an .fbx"
            )
        return "Mesh import: drop an .fbx file onto the window"

    def import_path(self, path: str | Path) -> ImportedMesh:
        """ load the largest mesh from an fbx file """
        mesh_path: Path = Path(path)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"mesh file not found: {mesh_path}")
        if not self.has_valid_suffix(mesh_path):
            raise ValueError(f"unsupported mesh format: {mesh_path.suffix}")
        name: str
        positions: tuple[float, ...]
        normals: tuple[float, ...]
        texcoords: tuple[float, ...]
        name, positions, normals, texcoords = load_fbx_via_bridge(mesh_path)
        if len(positions) < 9:
            raise ValueError(f"mesh has no triangles: {name}")
        return ImportedMesh(
            name=name if name else mesh_path.stem,
            source_path=str(mesh_path.resolve()),
            positions=positions,
            normals=normals,
            texcoords=texcoords,
        )


def mesh_id_for_import(imported: ImportedMesh) -> str:
    """ stable mesh table name for an imported asset """
    stem: str = Path(imported.source_path).stem
    safe_mesh: str = "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in imported.name
    )
    return f"fbx:{stem}:{safe_mesh}"
