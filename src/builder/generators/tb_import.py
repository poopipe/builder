"""generator: place meshes from a terrain builder object-import .txt"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from pathlib import Path

from pyray import Transform, Vector3

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    GeneratorSpec,
)
from builder.generators.orient import slots_from_transforms
from builder.generators.param_types import NameParam
from builder.heightfield.heightfield_types import (
    Heightfield,
    heightfield_layout,
    map_world_xz_to_hf_local,
)
from builder.modifiers.project_heightfield import resolve_heightfield_node
from builder.scene.orientation import quaternion_from_yaw_pitch_roll
from builder.tb_object import TbObject, parse_tb_object_import


@dataclass(frozen=True)
class TbImportParams:
    """recipe values for a tb object-import generator"""

    source_path: NameParam = NameParam("", label="Source")
    heightfield_node_id: NameParam = NameParam("", label="Heightfield")


def local_yaw_from_mapframe(mapframe_yaw_deg: float) -> float:
    """inverse of mapframe_yaw_deg for non-corner objects"""
    return -float(mapframe_yaw_deg)


def transform_from_tb_object(
    obj: TbObject,
    *,
    origin_x: float,
    origin_z: float,
    size_x: float,
    size_z: float,
) -> Transform:
    """mapframe object row → generator-local transform at y = 0"""
    local_x: float
    local_z: float
    local_x, local_z = map_world_xz_to_hf_local(
        obj.x,
        obj.y,
        origin_x=origin_x,
        origin_z=origin_z,
        size_x=size_x,
        size_z=size_z,
    )
    yaw: float = radians(local_yaw_from_mapframe(obj.yaw))
    pitch: float = radians(obj.pitch)
    roll: float = radians(obj.roll)
    scale: float = float(obj.scale)
    return Transform(
        Vector3(local_x, 0.0, local_z),
        quaternion_from_yaw_pitch_roll(yaw, pitch, roll),
        Vector3(scale, scale, scale),
    )


def tb_import_from_params(
    params: TbImportParams, context: BuildContext
) -> list[GeneratedSlot]:
    """build slots from a tb import file in heightfield mapframe"""
    hf_node = resolve_heightfield_node(
        context.nodes, params.heightfield_node_id.value
    )
    if hf_node is None or hf_node.heightfield is None:
        return []
    heightfield: Heightfield = hf_node.heightfield
    layout = heightfield_layout(heightfield)
    path_text: str = params.source_path.value.strip()
    if path_text == "":
        return []
    path: Path = Path(path_text)
    if not path.is_file():
        return []
    try:
        objects: list[TbObject] = parse_tb_object_import(path)
    except OSError:
        return []
    transforms: list[Transform] = [
        transform_from_tb_object(
            obj,
            origin_x=float(heightfield.origin_x.value),
            origin_z=float(heightfield.origin_z.value),
            size_x=layout.size_x,
            size_z=layout.size_z,
        )
        for obj in objects
    ]
    return slots_from_transforms(transforms)


tb_import_spec: GeneratorSpec = GeneratorSpec(
    kind="tb_import",
    label="TB Import",
    build_transforms=tb_import_from_params,
    params_type=TbImportParams,
)
