from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields, replace
from enum import IntEnum
from typing import Any, get_type_hints

from pyray import Transform, Vector3, Vector4

from builder.generators.generator_types import (
    Generator,
    MeshPattern,
    MeshSequenceMode,
    Modifier,
)
from builder.generators.param_types import (
    BoolParam,
    EnumParam,
    Float3Param,
    FloatParam,
    Int3Param,
    IntParam,
)
from builder.generators.registry import get_spec
from builder.heightfield.heightfield_types import Heightfield, HeightmapLayer
from builder.heightfield.heightmap_catalog import HeightmapAsset, HeightmapId
from builder.modifiers.registry import get_modifier_spec, known_modifier_kind
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind
from builder.scene.scene_types import MeshId, Node, NodeRole

scene_format_version: int = 7
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
    heightmaps: tuple[HeightmapAsset, ...]
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


def enum_from_name[E: IntEnum](enum_type: type[E], raw: Any, *, label: str) -> E:
    """decode an IntEnum member from its scene name"""
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a string name")
    try:
        return enum_type[raw]
    except KeyError as error:
        raise ValueError(f"unknown {label} {raw!r}") from error


def mesh_pattern_to_json(pattern: MeshPattern) -> dict[str, Any]:
    """encode a mesh pattern"""
    return {
        "mesh_ids": [mesh_id.name for mesh_id in pattern.mesh_ids],
        "mode": pattern.mode.name,
        "seed": pattern.seed,
    }


def mesh_pattern_from_json(data: dict[str, Any]) -> MeshPattern:
    """decode a mesh pattern"""
    ids_raw: Any = data.get("mesh_ids")
    if not isinstance(ids_raw, list):
        raise ValueError("generator.mesh_ids must be an array")
    mesh_ids: list[MeshId] = []
    entry: Any
    for entry in ids_raw:
        if not isinstance(entry, str) or entry == "":
            raise ValueError("generator.mesh_ids entries must be non-empty strings")
        mesh_ids.append(MeshId(entry))
    mode: MeshSequenceMode = enum_from_name(
        MeshSequenceMode, data.get("mode"), label="generator.mode"
    )
    seed_raw: Any = data.get("seed")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ValueError("generator.seed must be an integer")
    return MeshPattern(mesh_ids=tuple(mesh_ids), mode=mode, seed=seed_raw)


def vec3_to_json(param: Float3Param | Int3Param) -> dict[str, float | int]:
    """encode a Float3Param/Int3Param's components"""
    if isinstance(param, Int3Param):
        return {"x": int(param.x), "y": int(param.y), "z": int(param.z)}
    return {"x": float(param.x), "y": float(param.y), "z": float(param.z)}


def float3_from_json(data: Any, *, template: Float3Param) -> Float3Param:
    """decode float3 components; template keeps labels and clamp_range"""
    if not isinstance(data, dict):
        raise ValueError("float3 must be an object")
    x_raw: Any = data.get("x")
    y_raw: Any = data.get("y")
    z_raw: Any = data.get("z")
    for label, value in (("x", x_raw), ("y", y_raw), ("z", z_raw)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"float3.{label} must be a number")
    return replace(template, x=float(x_raw), y=float(y_raw), z=float(z_raw))


def int3_from_json(data: Any, *, template: Int3Param) -> Int3Param:
    """decode int3 components; template keeps labels and clamp_range"""
    if not isinstance(data, dict):
        raise ValueError("int3 must be an object")
    x_raw: Any = data.get("x")
    y_raw: Any = data.get("y")
    z_raw: Any = data.get("z")
    for label, value in (("x", x_raw), ("y", y_raw), ("z", z_raw)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"int3.{label} must be an integer")
    return replace(template, x=int(x_raw), y=int(y_raw), z=int(z_raw))


def modifier_to_json(modifier: Modifier) -> dict[str, Any]:
    """encode one modifier stack entry"""
    return {
        "kind": modifier.kind,
        "params": params_to_json(modifier.params),
        "enabled": modifier.enabled,
    }


def modifier_from_json(data: Any) -> Modifier | None:
    """decode one modifier; unknown kinds are dropped"""
    if not isinstance(data, dict):
        raise ValueError("modifier must be an object")
    kind: Any = data.get("kind")
    params_raw: Any = data.get("params")
    if not isinstance(kind, str) or kind == "":
        raise ValueError("modifier.kind must be a non-empty string")
    if not known_modifier_kind(kind):
        return None
    if not isinstance(params_raw, dict):
        raise ValueError("modifier.params must be an object")
    spec = get_modifier_spec(kind)
    params: Any = typed_params_from_json(spec.params_type, params_raw)
    enabled_raw: Any = data.get("enabled")
    if not isinstance(enabled_raw, bool):
        raise ValueError("modifier.enabled must be a boolean")
    return Modifier(kind=kind, params=params, enabled=enabled_raw)


def param_field_default(params_type: type, field_name: str) -> Any:
    """return the default wrapper instance for a params field"""
    for field in fields(params_type):
        if field.name != field_name:
            continue
        if field.default is not MISSING:
            return field.default
        if field.default_factory is not MISSING:
            return field.default_factory()
        raise ValueError(f"{params_type.__name__}.{field_name} has no default")
    raise KeyError(field_name)


def enum_member_from_json(enum_type: type[IntEnum], raw: Any) -> IntEnum:
    """decode an IntEnum member from its scene name"""
    return enum_from_name(enum_type, raw, label=enum_type.__name__)


def params_to_json(params: Any) -> dict[str, Any]:
    """encode a params dataclass to a json object (values only)"""
    out: dict[str, Any] = {}
    for field in fields(params):
        value: Any = getattr(params, field.name)
        if isinstance(value, Float3Param) or isinstance(value, Int3Param):
            out[field.name] = vec3_to_json(value)
        elif isinstance(value, EnumParam):
            out[field.name] = value.value.name
        elif isinstance(value, BoolParam):
            out[field.name] = 1 if value.value else 0
        elif isinstance(value, IntParam):
            out[field.name] = int(value.value)
        elif isinstance(value, FloatParam):
            out[field.name] = float(value.value)
        else:
            # skip non-wrapper fields (eg. heightfield.layers)
            continue
    return out


def typed_params_from_json(params_type: type, data: dict[str, Any]) -> Any:
    """build a params dataclass from json values, keeping field metadata"""
    hints: dict[str, Any] = get_type_hints(params_type)
    values: dict[str, Any] = {}
    for field in fields(params_type):
        if field.name not in data:
            continue
        raw: Any = data[field.name]
        annotation: Any = hints[field.name]
        default: Any = param_field_default(params_type, field.name)
        if annotation is Float3Param:
            values[field.name] = float3_from_json(raw, template=default)
            continue
        if annotation is Int3Param:
            values[field.name] = int3_from_json(raw, template=default)
            continue
        if annotation is EnumParam:
            enum_type: type[IntEnum] = type(default.value)
            values[field.name] = replace(
                default, value=enum_member_from_json(enum_type, raw)
            )
            continue
        if annotation is BoolParam:
            values[field.name] = replace(default, value=bool(int(raw)))
            continue
        if annotation is IntParam:
            values[field.name] = replace(default, value=int(raw))
            continue
        if annotation is FloatParam:
            values[field.name] = replace(default, value=float(raw))
            continue
        raise TypeError(
            f"unsupported param annotation for {field.name}: {annotation!r}"
        )
    return params_type(**values)


def generator_to_json(generator: Generator) -> dict[str, Any]:
    """encode a generator recipe"""
    payload: dict[str, Any] = {
        "kind": generator.kind,
        "meshes": mesh_pattern_to_json(generator.meshes),
        "params": params_to_json(generator.params),
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
    if not isinstance(meshes_raw, dict):
        raise ValueError("generator.meshes must be an object")
    pattern: MeshPattern = mesh_pattern_from_json(meshes_raw)
    point_meshes: MeshPattern | None = None
    point_raw: Any = data.get("point_meshes")
    if point_raw is not None:
        if not isinstance(point_raw, dict):
            raise ValueError("generator.point_meshes must be an object")
        point_meshes = mesh_pattern_from_json(point_raw)
    spec = get_spec(kind)
    params: Any = typed_params_from_json(spec.params_type, params_raw)
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
        "kind": asset.kind.name,
        "label": asset.label,
    }
    match asset.kind:
        case MeshAssetKind.fbx:
            if relative_source is None:
                raise ValueError(f"fbx asset {asset.mesh_id.name} has no source path")
            payload["source"] = relative_source
        case MeshAssetKind.builtin:
            pass
    return payload


def mesh_asset_kind_from_json(raw: Any) -> MeshAssetKind:
    """decode a mesh asset kind from its scene name"""
    return enum_from_name(MeshAssetKind, raw, label="mesh.kind")


def mesh_asset_from_json(data: Any) -> MeshAsset:
    """decode a catalog asset; source stays relative until load resolves it"""
    if not isinstance(data, dict):
        raise ValueError("mesh entry must be an object")
    mesh_name: Any = data.get("id")
    kind_raw: Any = data.get("kind")
    label: Any = data.get("label")
    if not isinstance(mesh_name, str) or mesh_name == "":
        raise ValueError("mesh.id must be a non-empty string")
    kind: MeshAssetKind = mesh_asset_kind_from_json(kind_raw)
    if not isinstance(label, str) or label == "":
        raise ValueError("mesh.label must be a non-empty string")
    source: str | None = None
    match kind:
        case MeshAssetKind.fbx:
            source_raw: Any = data.get("source")
            if not isinstance(source_raw, str) or source_raw == "":
                raise ValueError(f"fbx mesh {mesh_name} requires source")
            source = source_raw
        case MeshAssetKind.builtin:
            pass
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
    match node.role:
        case NodeRole.none:
            pass
        case _:
            payload["role"] = node.role.name
    if node.generator is not None:
        payload["generator"] = generator_to_json(node.generator)
    if node.heightfield is not None:
        payload["heightfield"] = heightfield_to_json(node.heightfield)
    return payload


def node_role_from_json(raw: Any) -> NodeRole:
    """decode a node role; missing/empty means none"""
    if raw is None or raw == "":
        return NodeRole.none
    return enum_from_name(NodeRole, raw, label="node.role")


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
    role: NodeRole = node_role_from_json(data.get("role", ""))
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
    heightfield: Heightfield | None = None
    if "heightfield" in data and data["heightfield"] is not None:
        heightfield = heightfield_from_json(data["heightfield"])
    if generator is not None and heightfield is not None:
        raise ValueError("node cannot have both generator and heightfield")
    return Node(
        id=node_id,
        parent_id=parent_id,
        transform=transform_from_json(data.get("transform")),
        mesh_id=mesh_id,
        generator=generator,
        heightfield=heightfield,
        name=name_raw,
        role=role,
    )


def heightfield_to_json(heightfield: Heightfield) -> dict[str, Any]:
    """encode a heightfield recipe"""
    return {
        "params": params_to_json(heightfield),
        "layers": [
            {
                "heightmap_id": layer.heightmap_id.name,
                "amplitude": float(layer.amplitude),
                "offset": float(layer.offset),
            }
            for layer in heightfield.layers
        ],
    }


def heightfield_from_json(data: Any) -> Heightfield:
    """decode a heightfield recipe"""
    if not isinstance(data, dict):
        raise ValueError("heightfield must be an object")
    params_raw: Any = data.get("params")
    if not isinstance(params_raw, dict):
        raise ValueError("heightfield.params must be an object")
    base: Heightfield = typed_params_from_json(Heightfield, params_raw)
    layers_raw: Any = data.get("layers", [])
    if not isinstance(layers_raw, list):
        raise ValueError("heightfield.layers must be an array")
    layers: list[HeightmapLayer] = []
    entry: Any
    for entry in layers_raw:
        if not isinstance(entry, dict):
            raise ValueError("heightfield layer must be an object")
        map_id: Any = entry.get("heightmap_id")
        if not isinstance(map_id, str) or map_id == "":
            raise ValueError("heightfield layer.heightmap_id must be a non-empty string")
        amplitude: Any = entry.get("amplitude", 1.0)
        offset: Any = entry.get("offset", 0.0)
        if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)):
            raise ValueError("heightfield layer.amplitude must be a number")
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise ValueError("heightfield layer.offset must be a number")
        layers.append(
            HeightmapLayer(
                heightmap_id=HeightmapId(map_id),
                amplitude=float(amplitude),
                offset=float(offset),
            )
        )
    return replace(base, layers=tuple(layers))


def heightmap_asset_to_json(
    asset: HeightmapAsset, relative_source: str
) -> dict[str, Any]:
    """encode a heightmap catalog entry"""
    return {
        "id": asset.heightmap_id.name,
        "label": asset.label,
        "source": relative_source,
    }


def heightmap_asset_from_json(data: Any) -> HeightmapAsset:
    """decode a heightmap catalog entry; source stays relative until load"""
    if not isinstance(data, dict):
        raise ValueError("heightmap entry must be an object")
    map_id: Any = data.get("id")
    label: Any = data.get("label")
    source: Any = data.get("source")
    if not isinstance(map_id, str) or map_id == "":
        raise ValueError("heightmap.id must be a non-empty string")
    if not isinstance(label, str) or label == "":
        raise ValueError("heightmap.label must be a non-empty string")
    if not isinstance(source, str) or source == "":
        raise ValueError("heightmap.source must be a non-empty string")
    return HeightmapAsset(
        heightmap_id=HeightmapId(map_id),
        label=label,
        source_path=source,
    )


def nodes_for_save(nodes: dict[str, Node]) -> list[Node]:
    """omit meshed children of live generator/heightfield groups; rebuilt on load

    nested group nodes are always kept, even when parented under a generator
    """
    saved: list[Node] = []
    node: Node
    for node in nodes.values():
        if node.parent_id is not None and node.mesh_id is not None:
            parent: Node | None = nodes.get(node.parent_id)
            if parent is not None and (
                parent.generator is not None or parent.heightfield is not None
            ):
                continue
        saved.append(node)
    return saved


def document_to_json(
    document: SceneDocument,
    relative_sources: dict[str, str],
    heightmap_sources: dict[str, str],
) -> dict[str, Any]:
    """encode a scene document; relative_sources maps mesh id -> relative path"""
    meshes_json: list[dict[str, Any]] = []
    asset: MeshAsset
    for asset in document.meshes:
        relative: str | None = relative_sources.get(asset.mesh_id.name)
        meshes_json.append(mesh_asset_to_json(asset, relative))
    heightmaps_json: list[dict[str, Any]] = []
    heightmap: HeightmapAsset
    for heightmap in document.heightmaps:
        relative_hm: str | None = heightmap_sources.get(heightmap.heightmap_id.name)
        if relative_hm is None:
            raise ValueError(
                f"heightmap '{heightmap.label}' has no relative source path"
            )
        heightmaps_json.append(heightmap_asset_to_json(heightmap, relative_hm))
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
        "heightmaps": heightmaps_json,
        "nodes": nodes_json,
    }


def document_from_json(data: Any) -> tuple[SceneDocument, list[str]]:
    """decode a scene document; returns (document, warnings)"""
    if not isinstance(data, dict):
        raise ValueError("scene root must be an object")
    version_raw: Any = data.get("version")
    if not isinstance(version_raw, int):
        raise ValueError("version must be an integer")
    if version_raw != scene_format_version:
        raise ValueError(
            f"unsupported scene version {version_raw}; "
            f"expected {scene_format_version}"
        )
    active_raw: Any = data.get("active_mesh_id")
    if not isinstance(active_raw, str) or active_raw == "":
        raise ValueError("active_mesh_id must be a non-empty string")
    ui_raw: Any = data.get("ui")
    if not isinstance(ui_raw, dict):
        raise ValueError("ui must be an object")
    side_open: Any = ui_raw.get("side_panel_open")
    meshes_open: Any = ui_raw.get("meshes_panel_open")
    outliner_open: Any = ui_raw.get("outliner_open")
    if (
        not isinstance(side_open, bool)
        or not isinstance(meshes_open, bool)
        or not isinstance(outliner_open, bool)
    ):
        raise ValueError("ui panel flags must be booleans")
    meshes_raw: Any = data.get("meshes")
    heightmaps_raw: Any = data.get("heightmaps")
    nodes_raw: Any = data.get("nodes")
    if not isinstance(meshes_raw, list) or not isinstance(nodes_raw, list):
        raise ValueError("meshes and nodes must be arrays")
    if not isinstance(heightmaps_raw, list):
        raise ValueError("heightmaps must be an array")
    meshes: list[MeshAsset] = [mesh_asset_from_json(entry) for entry in meshes_raw]
    heightmaps: list[HeightmapAsset] = [
        heightmap_asset_from_json(entry) for entry in heightmaps_raw
    ]
    nodes: list[Node] = [node_from_json(entry) for entry in nodes_raw]
    document: SceneDocument = SceneDocument(
        version=version_raw,
        active_mesh_id=MeshId(active_raw),
        ui=SceneUiState(
            side_panel_open=side_open,
            meshes_panel_open=meshes_open,
            outliner_open=outliner_open,
        ),
        meshes=tuple(meshes),
        heightmaps=tuple(heightmaps),
        nodes=tuple(nodes),
    )
    return document, []


def dumps_document(
    document: SceneDocument,
    relative_sources: dict[str, str],
    heightmap_sources: dict[str, str],
) -> str:
    """serialize a scene document to indented json text"""
    payload: dict[str, Any] = document_to_json(
        document, relative_sources, heightmap_sources
    )
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def loads_document(text: str) -> tuple[SceneDocument, list[str]]:
    """parse scene json text"""
    data: Any = json.loads(text)
    return document_from_json(data)
