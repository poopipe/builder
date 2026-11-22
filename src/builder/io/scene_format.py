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
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    HeightLayerSource,
    HeightLayerSourceType,
    Heightfield,
    PaintHeightSource,
    ProceduralHeightSource,
    ProceduralHeightType,
    SculptHeightMode,
    SculptHeightSource,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
    height_layer_source_type,
)
from builder.heightfield.heightmap_catalog import HeightmapAsset, HeightmapId
from builder.heightfield.regenerate import is_heightfield_tile_child
from builder.heightfield.vector.registry import (
    get_vector_shape_spec,
    known_vector_shape_kind,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorLayerRecipe,
    VectorShapeRecipe,
)
from builder.modifiers.registry import get_modifier_spec, known_modifier_kind
from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind
from builder.path.path_types import PathPointRecipe
from builder.scene.scene_types import MeshId, Node, NodeRole

scene_format_version: int = 26
scene_file_suffix: str = ".scene"


@dataclass(frozen=True)
class SceneUiState:
    """panel visibility persisted with a scene"""

    side_panel_open: bool
    meshes_panel_open: bool
    height_layers_panel_open: bool
    sculpt_panel_open: bool
    outliner_open: bool = True


@dataclass(frozen=True)
class SceneCameraState:
    """orbit camera pose and navigation speeds persisted with a scene"""

    yaw: float
    pitch: float
    distance: float
    target: Vector3
    zoom_speed: float
    move_speed: float
    view_distance: float


@dataclass(frozen=True)
class SceneDocument:
    """in-memory scene file contents"""

    version: int
    active_mesh_id: MeshId
    ui: SceneUiState
    camera: SceneCameraState
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
            # skip non-wrapper fields (eg. heightfield.base / edit_layers)
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


def vector_layer_to_json(_layer: VectorLayerRecipe) -> dict[str, Any]:
    """encode a vector layer recipe"""
    return {}


def vector_layer_from_json(data: Any) -> VectorLayerRecipe:
    """decode a vector layer recipe"""
    if not isinstance(data, dict):
        raise ValueError("vector_layer must be an object")
    return VectorLayerRecipe()


def vector_shape_to_json(shape: VectorShapeRecipe) -> dict[str, Any]:
    """encode a vector shape recipe"""
    return {
        "kind": shape.kind,
        "apply_mode": shape.apply_mode.name,
        "brush_kind": shape.brush_kind,
        "offset": float(shape.offset.value),
        "intensity": float(shape.intensity.value),
        "radius": float(shape.radius.value),
        "grade": float(shape.grade.value),
        "enabled": bool(shape.enabled),
        "params": params_to_json(shape.params),
    }


def path_point_to_json(point: PathPointRecipe) -> dict[str, Any]:
    """encode a path point recipe"""
    return {"width_scale": float(point.width_scale.value)}


def path_point_from_json(data: Any) -> PathPointRecipe:
    """decode a path point recipe"""
    if not isinstance(data, dict):
        raise ValueError("path_point must be an object")
    scale_raw: Any = data.get("width_scale", 1.0)
    if isinstance(scale_raw, bool) or not isinstance(scale_raw, (int, float)):
        raise ValueError("path_point.width_scale must be a number")
    return PathPointRecipe(
        width_scale=FloatParam(
            float(scale_raw),
            label="Width scale",
            step=0.05,
            minimum=0.05,
        )
    )


def vector_shape_from_json(data: Any) -> VectorShapeRecipe:
    """decode a vector shape recipe"""
    if not isinstance(data, dict):
        raise ValueError("vector_shape must be an object")
    kind_raw: Any = data.get("kind")
    if not isinstance(kind_raw, str) or kind_raw == "":
        raise ValueError("vector_shape.kind must be a non-empty string")
    if not known_vector_shape_kind(kind_raw):
        raise ValueError(f"unknown vector shape kind: {kind_raw}")
    apply_mode: VectorApplyMode = enum_from_name(
        VectorApplyMode, data.get("apply_mode"), label="vector_shape.apply_mode"
    )
    brush_kind_raw: Any = data.get("brush_kind", "add")
    if not isinstance(brush_kind_raw, str) or brush_kind_raw == "":
        raise ValueError("vector_shape.brush_kind must be a non-empty string")
    if brush_kind_raw == "subtract":
        raise ValueError(
            "vector_shape.brush_kind 'subtract' is removed; use add with negative offset"
        )
    offset_raw: Any = data.get("offset")
    intensity_raw: Any = data.get("intensity")
    radius_raw: Any = data.get("radius", 4.0)
    grade_raw: Any = data.get("grade")
    if isinstance(offset_raw, bool) or not isinstance(offset_raw, (int, float)):
        raise ValueError("vector_shape.offset must be a number")
    if isinstance(intensity_raw, bool) or not isinstance(
        intensity_raw, (int, float)
    ):
        raise ValueError("vector_shape.intensity must be a number")
    if isinstance(radius_raw, bool) or not isinstance(radius_raw, (int, float)):
        raise ValueError("vector_shape.radius must be a number")
    if isinstance(grade_raw, bool) or not isinstance(grade_raw, (int, float)):
        raise ValueError("vector_shape.grade must be a number")
    enabled_raw: Any = data.get("enabled", True)
    if not isinstance(enabled_raw, bool):
        raise ValueError("vector_shape.enabled must be a boolean")
    params_raw: Any = data.get("params", {})
    if not isinstance(params_raw, dict):
        raise ValueError("vector_shape.params must be an object")
    spec = get_vector_shape_spec(kind_raw)
    params: Any = typed_params_from_json(spec.params_type, params_raw)
    grade_value: float = max(0.0, min(1.0, float(grade_raw)))

    return VectorShapeRecipe(
        kind=kind_raw,
        apply_mode=apply_mode,
        brush_kind=brush_kind_raw,
        offset=FloatParam(float(offset_raw), label="Offset", step=0.25),
        intensity=FloatParam(
            float(intensity_raw),
            label="Intensity",
            step=0.05,
        ),
        radius=FloatParam(float(radius_raw), label="Radius", step=0.25, minimum=0.1),
        grade=FloatParam(
            grade_value,
            label="Grade",
            step=0.05,
            minimum=0.0,
            maximum=1.0,
        ),
        params=params,
        enabled=enabled_raw,
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
    if node.vector_layer is not None:
        payload["vector_layer"] = vector_layer_to_json(node.vector_layer)
    if node.vector_shape is not None:
        payload["vector_shape"] = vector_shape_to_json(node.vector_shape)
    if node.path_point is not None:
        payload["path_point"] = path_point_to_json(node.path_point)
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
    vector_layer: VectorLayerRecipe | None = None
    if "vector_layer" in data and data["vector_layer"] is not None:
        vector_layer = vector_layer_from_json(data["vector_layer"])
    vector_shape: VectorShapeRecipe | None = None
    if "vector_shape" in data and data["vector_shape"] is not None:
        vector_shape = vector_shape_from_json(data["vector_shape"])
    path_point: PathPointRecipe | None = None
    if "path_point" in data and data["path_point"] is not None:
        path_point = path_point_from_json(data["path_point"])
    recipe_count: int = sum(
        1
        for flag in (
            generator is not None,
            heightfield is not None,
            vector_layer is not None,
            vector_shape is not None,
        )
        if flag
    )
    if recipe_count > 1:
        raise ValueError(
            "node cannot combine generator/heightfield/vector_layer/vector_shape"
        )
    if path_point is not None and role is not NodeRole.path_point:
        raise ValueError("path_point recipe requires role path_point")
    if role is NodeRole.path_point and path_point is None:
        raise ValueError("path_point role requires a path_point recipe")
    if role in (
        NodeRole.path_handle_in,
        NodeRole.path_handle_out,
    ) and path_point is not None:
        raise ValueError("path handles cannot carry a path_point recipe")
    if path_point is not None and recipe_count > 0:
        raise ValueError(
            "path_point cannot combine with generator/heightfield/vector_layer/vector_shape"
        )
    return Node(
        id=node_id,
        parent_id=parent_id,
        transform=transform_from_json(data.get("transform")),
        mesh_id=mesh_id,
        generator=generator,
        heightfield=heightfield,
        vector_layer=vector_layer,
        vector_shape=vector_shape,
        path_point=path_point,
        name=name_raw,
        role=role,
    )


def height_layer_source_to_json(source: HeightLayerSource) -> dict[str, Any]:
    """encode a height or mask source"""
    source_type: HeightLayerSourceType = height_layer_source_type(source)
    match source:
        case CatalogHeightSource(heightmap_id=heightmap_id):
            return {
                "source_type": source_type.name,
                "heightmap_id": heightmap_id.name,
            }
        case PaintHeightSource(heightmap_id=heightmap_id):
            return {
                "source_type": source_type.name,
                "heightmap_id": heightmap_id.name,
            }
        case SculptHeightSource(
            heightmap_id=heightmap_id, mode=mode, coverage_id=coverage_id
        ):
            payload: dict[str, Any] = {
                "source_type": source_type.name,
                "heightmap_id": heightmap_id.name,
                "mode": mode.name,
            }
            if coverage_id is not None:
                payload["coverage_id"] = coverage_id.name
            return payload
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            return {
                "source_type": source_type.name,
                "procedural_type": procedural_type.name,
                "constant_value": float(constant_value),
            }
        case VectorHeightSource(
            layer_node_id=layer_node_id, heightmap_id=heightmap_id
        ):
            return {
                "source_type": source_type.name,
                "layer_node_id": layer_node_id,
                "heightmap_id": heightmap_id.name,
            }
        case VectorSculptHeightSource(
            layer_node_id=layer_node_id, heightmap_id=heightmap_id
        ):
            return {
                "source_type": source_type.name,
                "layer_node_id": layer_node_id,
                "heightmap_id": heightmap_id.name,
            }
        case VectorRoadHeightSource(
            layer_node_id=layer_node_id, heightmap_id=heightmap_id
        ):
            return {
                "source_type": source_type.name,
                "layer_node_id": layer_node_id,
                "heightmap_id": heightmap_id.name,
            }


def height_layer_source_from_json(data: Any, label: str) -> HeightLayerSource:
    """decode a height or mask source"""
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    type_raw: Any = data.get("source_type")
    source_type: HeightLayerSourceType = enum_from_name(
        HeightLayerSourceType, type_raw, label=f"{label}.source_type"
    )
    match source_type:
        case HeightLayerSourceType.catalog:
            map_id: Any = data.get("heightmap_id")
            if not isinstance(map_id, str) or map_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            return CatalogHeightSource(heightmap_id=HeightmapId(map_id))
        case HeightLayerSourceType.paint:
            paint_id: Any = data.get("heightmap_id")
            if not isinstance(paint_id, str) or paint_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            return PaintHeightSource(heightmap_id=HeightmapId(paint_id))
        case HeightLayerSourceType.sculpt:
            sculpt_id: Any = data.get("heightmap_id")
            if not isinstance(sculpt_id, str) or sculpt_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            mode: SculptHeightMode = enum_from_name(
                SculptHeightMode, data.get("mode"), label=f"{label}.mode"
            )
            coverage: HeightmapId | None = None
            cov_raw: Any = data.get("coverage_id")
            if cov_raw is not None:
                if not isinstance(cov_raw, str) or cov_raw == "":
                    raise ValueError(
                        f"{label}.coverage_id must be a non-empty string when set"
                    )
                coverage = HeightmapId(cov_raw)
            if mode is SculptHeightMode.absolute and coverage is None:
                raise ValueError(
                    f"{label}.coverage_id is required for absolute sculpt mode"
                )
            if mode is SculptHeightMode.delta:
                coverage = None
            return SculptHeightSource(
                heightmap_id=HeightmapId(sculpt_id),
                mode=mode,
                coverage_id=coverage,
            )
        case HeightLayerSourceType.procedural:
            proc_raw: Any = data.get("procedural_type")
            procedural_type: ProceduralHeightType = enum_from_name(
                ProceduralHeightType,
                proc_raw,
                label=f"{label}.procedural_type",
            )
            constant_raw: Any = data.get("constant_value", 0.0)
            if isinstance(constant_raw, bool) or not isinstance(
                constant_raw, (int, float)
            ):
                raise ValueError(f"{label}.constant_value must be a number")
            return ProceduralHeightSource(
                procedural_type=procedural_type,
                constant_value=float(constant_raw),
            )
        case HeightLayerSourceType.vector:
            layer_node_raw: Any = data.get("layer_node_id")
            if not isinstance(layer_node_raw, str) or layer_node_raw == "":
                raise ValueError(f"{label}.layer_node_id must be a non-empty string")
            vector_map_id: Any = data.get("heightmap_id")
            if not isinstance(vector_map_id, str) or vector_map_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            return VectorHeightSource(
                layer_node_id=layer_node_raw,
                heightmap_id=HeightmapId(vector_map_id),
            )
        case HeightLayerSourceType.vector_sculpt:
            sculpt_layer_raw: Any = data.get("layer_node_id")
            if not isinstance(sculpt_layer_raw, str) or sculpt_layer_raw == "":
                raise ValueError(f"{label}.layer_node_id must be a non-empty string")
            sculpt_map_id: Any = data.get("heightmap_id")
            if not isinstance(sculpt_map_id, str) or sculpt_map_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            return VectorSculptHeightSource(
                layer_node_id=sculpt_layer_raw,
                heightmap_id=HeightmapId(sculpt_map_id),
            )
        case HeightLayerSourceType.vector_road:
            road_layer_raw: Any = data.get("layer_node_id")
            if not isinstance(road_layer_raw, str) or road_layer_raw == "":
                raise ValueError(f"{label}.layer_node_id must be a non-empty string")
            road_map_id: Any = data.get("heightmap_id")
            if not isinstance(road_map_id, str) or road_map_id == "":
                raise ValueError(f"{label}.heightmap_id must be a non-empty string")
            return VectorRoadHeightSource(
                layer_node_id=road_layer_raw,
                heightmap_id=HeightmapId(road_map_id),
            )


def require_layer_float(data: dict[str, Any], key: str, default: float) -> float:
    """decode a numeric edit-layer float field"""
    raw: Any = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"heightfield edit layer.{key} must be a number")
    return float(raw)


def height_edit_layer_to_json(layer: HeightEditLayer) -> dict[str, Any]:
    """encode one edit layer"""
    payload: dict[str, Any] = {
        "name": layer.name,
        "enabled": bool(layer.enabled),
        "blend": layer.blend.name,
        "opacity": float(layer.opacity.value),
        "amplitude": float(layer.amplitude),
        "offset": float(layer.offset),
        "uv_scale_u": float(layer.uv_scale_u),
        "uv_scale_v": float(layer.uv_scale_v),
        "uv_offset_u": float(layer.uv_offset_u),
        "uv_offset_v": float(layer.uv_offset_v),
        "height": height_layer_source_to_json(layer.height),
    }
    if layer.mask is not None:
        payload["mask"] = height_layer_source_to_json(layer.mask)
    return payload


def height_edit_layer_from_json(data: Any) -> HeightEditLayer:
    """decode one edit layer"""
    if not isinstance(data, dict):
        raise ValueError("heightfield edit layer must be an object")
    name_raw: Any = data.get("name")
    if not isinstance(name_raw, str) or name_raw == "":
        raise ValueError("heightfield edit layer.name must be a non-empty string")
    enabled_raw: Any = data.get("enabled", True)
    if not isinstance(enabled_raw, bool):
        raise ValueError("heightfield edit layer.enabled must be a boolean")
    blend: HeightBlendMode = enum_from_name(
        HeightBlendMode, data.get("blend"), label="heightfield edit layer.blend"
    )
    opacity_raw: Any = data.get("opacity", 1.0)
    if isinstance(opacity_raw, bool) or not isinstance(opacity_raw, (int, float)):
        raise ValueError("heightfield edit layer.opacity must be a number")
    height_raw: Any = data.get("height")
    height: HeightLayerSource = height_layer_source_from_json(
        height_raw, "heightfield edit layer.height"
    )
    mask: HeightLayerSource | None = None
    if "mask" in data and data["mask"] is not None:
        mask = height_layer_source_from_json(
            data["mask"], "heightfield edit layer.mask"
        )
    return HeightEditLayer(
        name=name_raw,
        enabled=enabled_raw,
        blend=blend,
        opacity=FloatParam(
            float(opacity_raw),
            label="Opacity",
            step=0.05,
            minimum=0.0,
            maximum=1.0,
        ),
        height=height,
        mask=mask,
        amplitude=require_layer_float(data, "amplitude", 1.0),
        offset=require_layer_float(data, "offset", 0.0),
        uv_scale_u=require_layer_float(data, "uv_scale_u", 1.0),
        uv_scale_v=require_layer_float(data, "uv_scale_v", 1.0),
        uv_offset_u=require_layer_float(data, "uv_offset_u", 0.0),
        uv_offset_v=require_layer_float(data, "uv_offset_v", 0.0),
    )


def heightfield_to_json(heightfield: Heightfield) -> dict[str, Any]:
    """encode a heightfield recipe"""
    return {
        "params": params_to_json(heightfield),
        "base": None if heightfield.base is None else heightfield.base.name,
        "edit_layers": [
            height_edit_layer_to_json(layer) for layer in heightfield.edit_layers
        ],
    }


def heightfield_from_json(data: Any) -> Heightfield:
    """decode a heightfield recipe"""
    if not isinstance(data, dict):
        raise ValueError("heightfield must be an object")
    params_raw: Any = data.get("params")
    if not isinstance(params_raw, dict):
        raise ValueError("heightfield.params must be an object")
    recipe: Heightfield = typed_params_from_json(Heightfield, params_raw)
    base_raw: Any = data.get("base")
    base: HeightmapId | None
    if base_raw is None:
        base = None
    elif isinstance(base_raw, str) and base_raw != "":
        base = HeightmapId(base_raw)
    else:
        raise ValueError("heightfield.base must be a non-empty string or null")
    layers_raw: Any = data.get("edit_layers", [])
    if not isinstance(layers_raw, list):
        raise ValueError("heightfield.edit_layers must be an array")
    edit_layers: list[HeightEditLayer] = [
        height_edit_layer_from_json(entry) for entry in layers_raw
    ]
    return replace(recipe, base=base, edit_layers=tuple(edit_layers))


def heightmap_asset_to_json(
    asset: HeightmapAsset,
    relative_source: str,
    relative_cache: str,
) -> dict[str, Any]:
    """encode a heightmap catalog entry"""
    return {
        "id": asset.heightmap_id.name,
        "label": asset.label,
        "source": relative_source,
        "cache": relative_cache,
        "sample_type": asset.sample_type.name,
        "width": asset.width,
        "height": asset.height,
    }


def heightmap_asset_from_json(data: Any) -> HeightmapAsset:
    """decode a heightmap catalog entry; paths stay relative until load"""
    if not isinstance(data, dict):
        raise ValueError("heightmap entry must be an object")
    map_id: Any = data.get("id")
    label: Any = data.get("label")
    source: Any = data.get("source")
    cache: Any = data.get("cache")
    sample_raw: Any = data.get("sample_type")
    width_raw: Any = data.get("width")
    height_raw: Any = data.get("height")
    if not isinstance(map_id, str) or map_id == "":
        raise ValueError("heightmap.id must be a non-empty string")
    if not isinstance(label, str) or label == "":
        raise ValueError("heightmap.label must be a non-empty string")
    if not isinstance(source, str) or source == "":
        raise ValueError("heightmap.source must be a non-empty string")
    if not isinstance(cache, str) or cache == "":
        raise ValueError("heightmap.cache must be a non-empty string")
    if not isinstance(sample_raw, str) or sample_raw == "":
        raise ValueError("heightmap.sample_type must be a non-empty string")
    if not isinstance(width_raw, int) or width_raw <= 0:
        raise ValueError("heightmap.width must be a positive integer")
    if not isinstance(height_raw, int) or height_raw <= 0:
        raise ValueError("heightmap.height must be a positive integer")
    try:
        sample_type: HeightSampleType = HeightSampleType[sample_raw]
    except KeyError as exc:
        raise ValueError(
            f"heightmap.sample_type must be a HeightSampleType name, got {sample_raw!r}"
        ) from exc
    return HeightmapAsset(
        heightmap_id=HeightmapId(map_id),
        label=label,
        source_path=source,
        cache_path=cache,
        sample_type=sample_type,
        width=width_raw,
        height=height_raw,
    )


def nodes_for_save(nodes: dict[str, Node]) -> list[Node]:
    """omit regenerated children of live generator/heightfield groups; rebuilt on load

    nested group nodes are always kept, even when parented under a generator.
    heightfield tile markers (mesh-less) are omitted the same way as old tile meshes
    """
    saved: list[Node] = []
    node: Node
    for node in nodes.values():
        if node.parent_id is None:
            saved.append(node)
            continue
        parent: Node | None = nodes.get(node.parent_id)
        if parent is None:
            saved.append(node)
            continue
        if parent.heightfield is not None and is_heightfield_tile_child(
            parent.id, node
        ):
            continue
        if parent.generator is not None and node.mesh_id is not None:
            continue
        saved.append(node)
    return saved


def document_to_json(
    document: SceneDocument,
    relative_sources: dict[str, str],
    heightmap_sources: dict[str, str],
    heightmap_caches: dict[str, str],
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
        relative_cache: str | None = heightmap_caches.get(heightmap.heightmap_id.name)
        if relative_hm is None:
            raise ValueError(
                f"heightmap '{heightmap.label}' has no relative source path"
            )
        if relative_cache is None:
            raise ValueError(
                f"heightmap '{heightmap.label}' has no relative cache path"
            )
        heightmaps_json.append(
            heightmap_asset_to_json(heightmap, relative_hm, relative_cache)
        )
    nodes_json: list[dict[str, Any]] = [node_to_json(node) for node in document.nodes]
    return {
        "version": document.version,
        "active_mesh_id": document.active_mesh_id.name,
        "ui": {
            "side_panel_open": document.ui.side_panel_open,
            "meshes_panel_open": document.ui.meshes_panel_open,
            "height_layers_panel_open": document.ui.height_layers_panel_open,
            "sculpt_panel_open": document.ui.sculpt_panel_open,
            "outliner_open": document.ui.outliner_open,
        },
        "camera": {
            "yaw": float(document.camera.yaw),
            "pitch": float(document.camera.pitch),
            "distance": float(document.camera.distance),
            "target": vec3_to_list(document.camera.target),
            "zoom_speed": float(document.camera.zoom_speed),
            "move_speed": float(document.camera.move_speed),
            "view_distance": float(document.camera.view_distance),
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
    height_layers_open: Any = ui_raw.get("height_layers_panel_open")
    sculpt_open: Any = ui_raw.get("sculpt_panel_open")
    outliner_open: Any = ui_raw.get("outliner_open")
    if (
        not isinstance(side_open, bool)
        or not isinstance(meshes_open, bool)
        or not isinstance(height_layers_open, bool)
        or not isinstance(sculpt_open, bool)
        or not isinstance(outliner_open, bool)
    ):
        raise ValueError("ui panel flags must be booleans")
    camera_raw: Any = data.get("camera")
    if not isinstance(camera_raw, dict):
        raise ValueError("camera must be an object")
    yaw_raw: Any = camera_raw.get("yaw")
    pitch_raw: Any = camera_raw.get("pitch")
    distance_raw: Any = camera_raw.get("distance")
    zoom_speed_raw: Any = camera_raw.get("zoom_speed")
    move_speed_raw: Any = camera_raw.get("move_speed")
    view_distance_raw: Any = camera_raw.get("view_distance")
    if (
        not isinstance(yaw_raw, (int, float))
        or not isinstance(pitch_raw, (int, float))
        or not isinstance(distance_raw, (int, float))
        or isinstance(yaw_raw, bool)
        or isinstance(pitch_raw, bool)
        or isinstance(distance_raw, bool)
    ):
        raise ValueError("camera yaw, pitch, and distance must be numbers")
    if float(distance_raw) <= 0.0:
        raise ValueError("camera distance must be positive")
    if (
        isinstance(zoom_speed_raw, bool)
        or not isinstance(zoom_speed_raw, (int, float))
        or isinstance(move_speed_raw, bool)
        or not isinstance(move_speed_raw, (int, float))
        or isinstance(view_distance_raw, bool)
        or not isinstance(view_distance_raw, (int, float))
    ):
        raise ValueError(
            "camera zoom_speed, move_speed, and view_distance must be numbers"
        )
    camera_target: Vector3 = list_to_vec3(camera_raw.get("target"), "camera.target")
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
            height_layers_panel_open=height_layers_open,
            sculpt_panel_open=sculpt_open,
            outliner_open=outliner_open,
        ),
        camera=SceneCameraState(
            yaw=float(yaw_raw),
            pitch=float(pitch_raw),
            distance=float(distance_raw),
            target=camera_target,
            zoom_speed=float(zoom_speed_raw),
            move_speed=float(move_speed_raw),
            view_distance=float(view_distance_raw),
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
    heightmap_caches: dict[str, str],
) -> str:
    """serialize a scene document to indented json text"""
    payload: dict[str, Any] = document_to_json(
        document, relative_sources, heightmap_sources, heightmap_caches
    )
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def loads_document(text: str) -> tuple[SceneDocument, list[str]]:
    """parse scene json text"""
    data: Any = json.loads(text)
    return document_from_json(data)
