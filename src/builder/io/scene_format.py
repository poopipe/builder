from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pyray import Transform, Vector3, Vector4

from builder.generators.generator_types import Generator, ParamMap, ParamValue
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind, MeshCatalog
from builder.scene.scene_types import MeshId, Node

scene_format_version: int = 1
scene_file_suffix: str = ".scene"


@dataclass(frozen=True)
class SceneUiState:
    """panel visibility persisted with a scene"""

    side_panel_open: bool
    meshes_panel_open: bool


@dataclass(frozen=True)
class SceneDocument:
    """in-memory scene file contents"""

    version: int
    active_mesh_id: MeshId
    ui: SceneUiState
    meshes: tuple[MeshAsset, ...]
    nodes: tuple[Node, ...]


def vec3_to_list(value: Vector3) -> list[float]:
    """encode a Vector3 as a json array"""
    return [float(value.x), float(value.y), float(value.z)]


def vec4_to_list(value: Vector4) -> list[float]:
    """encode a Vector4 / quaternion as a json array"""
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def list_to_vec3(values: Any, label: str) -> Vector3:
    """decode a json array into a Vector3"""
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label} must be a list of 3 numbers")
    return Vector3(float(values[0]), float(values[1]), float(values[2]))


def list_to_vec4(values: Any, label: str) -> Vector4:
    """decode a json array into a Vector4"""
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"{label} must be a list of 4 numbers")
    return Vector4(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def transform_to_json(transform: Transform) -> dict[str, list[float]]:
    """encode a raylib Transform"""
    return {
        "translation": vec3_to_list(transform.translation),
        "rotation": vec4_to_list(transform.rotation),
        "scale": vec3_to_list(transform.scale),
    }


def transform_from_json(data: Any) -> Transform:
    """decode a raylib Transform"""
    if not isinstance(data, dict):
        raise ValueError("transform must be an object")
    return Transform(
        list_to_vec3(data.get("translation"), "translation"),
        list_to_vec4(data.get("rotation"), "rotation"),
        list_to_vec3(data.get("scale"), "scale"),
    )


def generator_to_json(generator: Generator) -> dict[str, Any]:
    """encode a generator recipe"""
    return {
        "kind": generator.kind,
        "mesh_id": generator.mesh_id.name,
        "params": dict(generator.params),
    }


def generator_from_json(data: Any) -> Generator:
    """decode a generator recipe"""
    if not isinstance(data, dict):
        raise ValueError("generator must be an object")
    kind: Any = data.get("kind")
    mesh_name: Any = data.get("mesh_id")
    params_raw: Any = data.get("params")
    if not isinstance(kind, str) or kind == "":
        raise ValueError("generator.kind must be a non-empty string")
    if not isinstance(mesh_name, str) or mesh_name == "":
        raise ValueError("generator.mesh_id must be a non-empty string")
    if not isinstance(params_raw, dict):
        raise ValueError("generator.params must be an object")
    params: ParamMap = {}
    key: Any
    value: Any
    for key, value in params_raw.items():
        if not isinstance(key, str):
            raise ValueError("generator.params keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"generator.params[{key!r}] must be a number")
        param_value: ParamValue = int(value) if isinstance(value, int) else float(value)
        params[key] = param_value
    return Generator(kind=kind, mesh_id=MeshId(mesh_name), params=params)


def mesh_asset_to_json(asset: MeshAsset, relative_source: str | None) -> dict[str, Any]:
    """encode a catalog asset; relative_source is used for fbx entries"""
    payload: dict[str, Any] = {
        "id": asset.mesh_id.name,
        "kind": asset.kind,
        "label": asset.label,
    }
    if asset.kind == "fbx":
        if relative_source is None:
            raise ValueError(f"fbx asset {asset.mesh_id.name} has no source path")
        payload["source"] = relative_source
    return payload


def mesh_asset_from_json(data: Any) -> MeshAsset:
    """decode a catalog asset; source stays relative until load resolves it"""
    if not isinstance(data, dict):
        raise ValueError("mesh entry must be an object")
    mesh_name: Any = data.get("id")
    kind_raw: Any = data.get("kind")
    label: Any = data.get("label")
    if not isinstance(mesh_name, str) or mesh_name == "":
        raise ValueError("mesh.id must be a non-empty string")
    if kind_raw not in ("builtin", "fbx"):
        raise ValueError("mesh.kind must be 'builtin' or 'fbx'")
    kind: MeshAssetKind = kind_raw
    if not isinstance(label, str) or label == "":
        raise ValueError("mesh.label must be a non-empty string")
    source: str | None = None
    if kind == "fbx":
        source_raw: Any = data.get("source")
        if not isinstance(source_raw, str) or source_raw == "":
            raise ValueError(f"fbx mesh {mesh_name} requires source")
        source = source_raw
    return MeshAsset(
        mesh_id=MeshId(mesh_name),
        label=label,
        kind=kind,
        source_path=source,
    )


def node_to_json(node: Node) -> dict[str, Any]:
    """encode one scene node"""
    payload: dict[str, Any] = {
        "id": node.id,
        "parent_id": node.parent_id,
        "transform": transform_to_json(node.transform),
        "mesh_id": None if node.mesh_id is None else node.mesh_id.name,
    }
    if node.generator is not None:
        payload["generator"] = generator_to_json(node.generator)
    return payload


def node_from_json(data: Any) -> Node:
    """decode one scene node"""
    if not isinstance(data, dict):
        raise ValueError("node must be an object")
    node_id: Any = data.get("id")
    parent_id: Any = data.get("parent_id")
    mesh_name: Any = data.get("mesh_id")
    if not isinstance(node_id, str) or node_id == "":
        raise ValueError("node.id must be a non-empty string")
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError("node.parent_id must be a string or null")
    mesh_id: MeshId | None
    if mesh_name is None:
        mesh_id = None
    elif isinstance(mesh_name, str) and mesh_name != "":
        mesh_id = MeshId(mesh_name)
    else:
        raise ValueError("node.mesh_id must be a string or null")
    generator: Generator | None = None
    if "generator" in data and data["generator"] is not None:
        generator = generator_from_json(data["generator"])
    return Node(
        id=node_id,
        parent_id=parent_id,
        transform=transform_from_json(data.get("transform")),
        mesh_id=mesh_id,
        generator=generator,
    )


def nodes_for_save(nodes: dict[str, Node]) -> list[Node]:
    """omit children of live generator groups; those are rebuilt on load"""
    saved: list[Node] = []
    node: Node
    for node in nodes.values():
        if node.parent_id is not None:
            parent: Node | None = nodes.get(node.parent_id)
            if parent is not None and parent.generator is not None:
                continue
        saved.append(node)
    return saved


def document_to_json(
    document: SceneDocument,
    relative_sources: dict[str, str],
) -> dict[str, Any]:
    """encode a scene document; relative_sources maps mesh id -> relative path"""
    meshes_json: list[dict[str, Any]] = []
    asset: MeshAsset
    for asset in document.meshes:
        relative: str | None = relative_sources.get(asset.mesh_id.name)
        meshes_json.append(mesh_asset_to_json(asset, relative))
    nodes_json: list[dict[str, Any]] = [node_to_json(node) for node in document.nodes]
    return {
        "version": document.version,
        "active_mesh_id": document.active_mesh_id.name,
        "ui": {
            "side_panel_open": document.ui.side_panel_open,
            "meshes_panel_open": document.ui.meshes_panel_open,
        },
        "meshes": meshes_json,
        "nodes": nodes_json,
    }


def document_from_json(data: Any) -> tuple[SceneDocument, list[str]]:
    """decode a scene document; returns (document, warnings)"""
    warnings: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("scene root must be an object")
    version_raw: Any = data.get("version", scene_format_version)
    if not isinstance(version_raw, int):
        raise ValueError("version must be an integer")
    version: int = version_raw
    if version != scene_format_version:
        warnings.append(
            f"scene version {version} differs from supported "
            f"{scene_format_version}; loading anyway"
        )
    active_raw: Any = data.get("active_mesh_id", "cube")
    if not isinstance(active_raw, str) or active_raw == "":
        raise ValueError("active_mesh_id must be a non-empty string")
    ui_raw: Any = data.get("ui", {})
    if not isinstance(ui_raw, dict):
        raise ValueError("ui must be an object")
    side_open: Any = ui_raw.get("side_panel_open", True)
    meshes_open: Any = ui_raw.get("meshes_panel_open", True)
    if not isinstance(side_open, bool) or not isinstance(meshes_open, bool):
        raise ValueError("ui panel flags must be booleans")
    meshes_raw: Any = data.get("meshes", [])
    nodes_raw: Any = data.get("nodes", [])
    if not isinstance(meshes_raw, list) or not isinstance(nodes_raw, list):
        raise ValueError("meshes and nodes must be arrays")
    meshes: list[MeshAsset] = [mesh_asset_from_json(entry) for entry in meshes_raw]
    nodes: list[Node] = [node_from_json(entry) for entry in nodes_raw]
    document: SceneDocument = SceneDocument(
        version=version,
        active_mesh_id=MeshId(active_raw),
        ui=SceneUiState(
            side_panel_open=side_open,
            meshes_panel_open=meshes_open,
        ),
        meshes=tuple(meshes),
        nodes=tuple(nodes),
    )
    return document, warnings


def dumps_document(
    document: SceneDocument,
    relative_sources: dict[str, str],
) -> str:
    """serialize a scene document to indented json text"""
    payload: dict[str, Any] = document_to_json(document, relative_sources)
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def loads_document(text: str) -> tuple[SceneDocument, list[str]]:
    """parse scene json text"""
    data: Any = json.loads(text)
    return document_from_json(data)


def catalog_to_assets(catalog: MeshCatalog) -> tuple[MeshAsset, ...]:
    """snapshot catalog entries for saving"""
    return tuple(catalog.entries.values())
