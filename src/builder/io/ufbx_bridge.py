"""ctypes loader for the ufbx_bridge DLL"""

from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    cdll,
)
from pathlib import Path
from typing import Any


class BridgeMesh(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("positions", POINTER(c_float)),
        ("normals", POINTER(c_float)),
        ("texcoords", POINTER(c_float)),
        ("vertex_count", c_int),
        ("triangle_count", c_int),
    ]


def default_dll_path() -> Path:
    """repo-relative path produced by ufbx_bridge/build.cmd"""
    repo_root: Path = Path(__file__).resolve().parents[3]
    return repo_root / "ufbx_bridge" / "build" / "ufbx_bridge.dll"


def load_bridge_library(dll_path: Path | None = None) -> ctypes.CDLL:
    """load ufbx_bridge.dll; raises FileNotFoundError if missing"""
    path: Path = dll_path if dll_path is not None else default_dll_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"ufbx_bridge.dll not found at {path}. "
            "Run ufbx_bridge\\build.cmd from a VS x64 Native Tools (or gcc) cmd prompt."
        )
    lib: ctypes.CDLL = cdll.LoadLibrary(str(path))
    lib.bridge_load_fbx.argtypes = [
        c_char_p,
        POINTER(BridgeMesh),
        c_char_p,
        c_int,
    ]
    lib.bridge_load_fbx.restype = c_int
    lib.bridge_free_mesh.argtypes = [POINTER(BridgeMesh)]
    lib.bridge_free_mesh.restype = None
    return lib


def floats_from_pointer(ptr: Any, count: int) -> tuple[float, ...]:
    """copy count floats from a C float* into a Python tuple"""
    if not ptr or count <= 0:
        return ()
    index: int
    return tuple(float(ptr[index]) for index in range(count))


def load_fbx_via_bridge(
    path: str | Path,
    lib: ctypes.CDLL | None = None,
) -> tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """
    load the densest mesh from an fbx via the ufbx bridge

    returns (name, positions, normals, texcoords) as flat float tuples
    """
    bridge: ctypes.CDLL = lib if lib is not None else load_bridge_library()
    mesh: BridgeMesh = BridgeMesh()
    err_cap: int = 1024
    err_buf: Any = ctypes.create_string_buffer(err_cap)
    path_bytes: bytes = str(Path(path)).encode("utf-8")
    rc: int = int(
        bridge.bridge_load_fbx(path_bytes, byref(mesh), err_buf, err_cap)
    )
    if rc != 0:
        message: str = err_buf.value.decode("utf-8", errors="replace") or "bridge_load_fbx failed"
        raise RuntimeError(message)
    try:
        name_raw: bytes | None = mesh.name
        name: str = (
            name_raw.decode("utf-8", errors="replace") if name_raw else Path(path).stem
        )
        vertex_count: int = int(mesh.vertex_count)
        positions: tuple[float, ...] = floats_from_pointer(
            mesh.positions, vertex_count * 3
        )
        normals: tuple[float, ...] = floats_from_pointer(
            mesh.normals, vertex_count * 3
        )
        texcoords: tuple[float, ...] = floats_from_pointer(
            mesh.texcoords, vertex_count * 2
        )
        return name, positions, normals, texcoords
    finally:
        bridge.bridge_free_mesh(byref(mesh))
