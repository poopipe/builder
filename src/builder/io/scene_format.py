from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pyray import Transform, Vector3, Vector4

from builder.generators.generator_types import (
    Generator,
    MeshPattern,
    MeshSequenceMode,
    Modifier,
    ParamMap,
    ParamValue,
)
from builder.modifiers.registry import known_modifier_kind
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind
from builder.scene.scene_types import MeshId, Node

scene_format_version: int = 4
scene_file_suffix: str = ".scene"


@dataclass(frozen=True)
class SceneUiState:
    """panel visibility persisted with a scene"""

    side_panel_open: bool
    meshes_panel_open: bool
    outliner_open: bool = True


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


def mesh_pattern_to_json(pattern: MeshPattern) -> dict[str, Any]:
    """encode a mesh pattern"""
    return {
        "mesh_ids": [mesh_id.name for mesh_id in pattern.mesh_ids],
        "mode": pattern.mode,
        "seed": pattern.seed,
    }


def mesh_pattern_from_json(data: dict[str, Any]) -> MeshPattern:
    """decode a mesh pattern, accepting the legacy single 'mesh_id' string"""
    ids_raw: Any = data.get("mesh_ids")
    if ids_raw is None:
        legacy: Any = data.get("mesh_id")
        if not isinstance(legacy, str) or legacy == "":
            raise ValueError("generator requires 'mesh_ids' or legacy 'mesh_id'")
        return MeshPattern(mesh_ids=(MeshId(legacy),), mode="repeat", seed=0)
    if not isinstance(ids_raw, list):
        raise ValueError("generator.mesh_ids must be an array")
    mesh_ids: list[MeshId] = []
    entry: Any
    for entry in ids_raw:
        if not isinstance(entry, str) or entry == "":
            raise ValueError("generator.mesh_ids entries must be non-empty strings")
        mesh_ids.append(MeshId(entry))
    mode_raw: Any = data.get("mode", "repeat")
    if mode_raw not in ("repeat", "pingpong", "random"):
        raise ValueError("generator.mode must be repeat, pingpong, or random")
    mode: MeshSequenceMode = mode_raw
    seed_raw: Any = data.get("seed", 0)
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ValueError("generator.seed must be an integer")
    return MeshPattern(mesh_ids=tuple(mesh_ids), mode=mode, seed=seed_raw)


def modifier_to_json(modifier: Modifier) -> dict[str, Any]:
    """encode one modifier stack entry"""
    return {
        "kind": modifier.kind,
        "params": dict(modifier.params),
        "enabled": modifier.enabled,
    }


def modifier_from_json(data: Any) -> Modifier | None:
    """decode one modifier; unknown kinds are dropped"""
    if not isinstance(data, dict):
        raise ValueError("modifier must be an object")
    kind: Any = data.get("kind")
    params_raw: Any = data.get("params", {})
    if not isinstance(kind, str) or kind == "":
        raise ValueError("modifier.kind must be a non-empty string")
    if not known_modifier_kind(kind):
        return None
    if not isinstance(params_raw, dict):
        raise ValueError("modifier.params must be an object")
    params: ParamMap = {}
    key: Any
    value: Any
    for key, value in params_raw.items():
        if not isinstance(key, str):
            raise ValueError("modifier.params keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"modifier.params[{key!r}] must be a number")
        params[key] = int(value) if isinstance(value, int) else float(value)
    enabled_raw: Any = data.get("enabled", True)
    if not isinstance(enabled_raw, bool):
        raise ValueError("modifier.enabled must be a boolean")
    return Modifier(kind=kind, params=params, enabled=enabled_raw)


def generator_to_json(generator: Generator) -> dict[str, Any]:
    """encode a generator recipe"""
    payload: dict[str, Any] = {
        "kind": generator.kind,
        "meshes": mesh_pattern_to_json(generator.meshes),
        "params": dict(generator.params),
    }
    if generator.point_meshes is not None:
        payload["point_meshes"] = mesh_pattern_to_json(generator.point_meshes)
    if generator.modifiers:
        payload["modifiers"] = [
            modifier_to_json(modifier) for modifier in generator.modifiers
        ]
    return payload


def generator_from_json(data: Any) -> Generator:
    """decode a generator recipe"""
    if not isinstance(data, dict):
        raise ValueError("generator must be an object")
    kind: Any = data.get("kind")
    params_raw: Any = data.get("params")
    if not isinstance(kind, str) or kind == "":
        raise ValueError("generator.kind must be a non-empty string")
    if not isinstance(params_raw, dict):
        raise ValueError("generator.params must be an object")
    meshes_raw: Any = data.get("meshes")
    # v2 nests the pattern under "meshes"; v1 kept a flat "mesh_id" string
    pattern: MeshPattern = mesh_pattern_from_json(
        meshes_raw if isinstance(meshes_raw, dict) else data
    )
    point_meshes: MeshPattern | None = None
    point_raw: Any = data.get("point_meshes")
    if isinstance(point_raw, dict):
        point_meshes = mesh_pattern_from_json(point_raw)
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
    # migrate renamed generator kinds from older scene files
    if kind == "radial_grid":
        kind = "radial"
    elif kind == "ngon_grid":
        kind = "ngon"
    # migrate radial face_center bool into facing enum
    if kind == "radial" and "facing" not in params and "face_center" in params:
        params["facing"] = 1 if int(params["face_center"]) else 0
    modifiers: list[Modifier] = []
    modifiers_raw: Any = data.get("modifiers", [])
    if modifiers_raw is None:
        modifiers_raw = []
    if not isinstance(modifiers_raw, list):
        raise ValueError("generator.modifiers must be an array")
    entry: Any
    for entry in modifiers_raw:
        decoded: Modifier | None = modifier_from_json(entry)
        if decoded is not None:
            modifiers.append(decoded)
    return Generator(
        kind=kind,
        meshes=pattern,
        params=params,
        point_meshes=point_meshes,
        modifiers=tuple(modifiers),
    )


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
    if node.name != "":
        payload["name"] = node.name
    if node.role != "":
        payload["role"] = node.role
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
    name_raw: Any = data.get("name", "")
    if not isinstance(name_raw, str):
        raise ValueError("node.name must be a string")
    role_raw: Any = data.get("role", "")
    if not isinstance(role_raw, str):
        raise ValueError("node.role must be a string")
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
        name=name_raw,
        role=role_raw,
    )


def nodes_for_save(nodes: dict[str, Node]) -> list[Node]:
    """omit meshed children of live generator groups; those are rebuilt on load

    nested group nodes are always kept, even when parented under a generator
    """
    saved: list[Node] = []
    node: Node
    for node in nodes.values():
        if node.parent_id is not None and node.mesh_id is not None:
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
            "outliner_open": document.ui.outliner_open,
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
    outliner_open: Any = ui_raw.get("outliner_open", True)
    if (
        not isinstance(side_open, bool)
        or not isinstance(meshes_open, bool)
        or not isinstance(outliner_open, bool)
    ):
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
            outliner_open=outliner_open,
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
