"""project generator slots onto a heightfield surface"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

from pyray import (
    Matrix,
    Transform,
    Vector3,
    matrix_invert,
    matrix_multiply,
    quaternion_invert,
    quaternion_multiply,
    vector3_normalize,
    vector3_transform,
)

from builder.generators.generator_types import BuildContext, GeneratedSlot
from builder.generators.param_types import BoolParam, FloatParam, NameParam
from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height
from builder.heightfield.vector.vector_types import world_xz_to_uv
from builder.modifiers.modifier_types import ModifierSpec
from builder.scene.orientation import quaternion_align_y
from builder.scene.scene_types import Node, Quaternion
from builder.scene.transforms import (
    matrix_translation,
    transform_direction,
    transform_matrix,
    world_matrix,
    world_rotation,
)


@dataclass(frozen=True)
class ProjectHeightfieldParams:
    """recipe values for project onto heightfield"""

    offset: FloatParam = FloatParam(
        0.0, label="Offset", step=0.25, minimum=-100.0, maximum=100.0
    )
    heightfield_node_id: NameParam = NameParam("", label="Heightfield")
    align_to_normal: BoolParam = BoolParam(False, label="Align to normal")


def heightfield_node_ids(nodes: Mapping[str, Node]) -> tuple[str, ...]:
    """sorted ids of nodes that own a heightfield recipe"""
    return tuple(
        sorted(
            node.id for node in nodes.values() if node.heightfield is not None
        )
    )


def heightfield_display_label(nodes: Mapping[str, Node], node_id: str) -> str:
    """spinner label for a heightfield node id"""
    node: Node | None = nodes.get(node_id)
    if node is None:
        return node_id
    if node.name != "":
        return node.name
    return node.id


def resolve_heightfield_node(
    nodes: Mapping[str, Node], heightfield_node_id: str
) -> Node | None:
    """pick target heightfield; empty/invalid only auto-selects when exactly one exists"""
    if heightfield_node_id != "":
        node: Node | None = nodes.get(heightfield_node_id)
        if node is not None and node.heightfield is not None:
            return node
    ids: tuple[str, ...] = heightfield_node_ids(nodes)
    if len(ids) == 1:
        return nodes[ids[0]]
    return None


def cycle_heightfield_id(
    nodes: Mapping[str, Node], current_id: str, delta: int
) -> str | None:
    """step through sorted heightfield ids; None when the scene has none"""
    ids: tuple[str, ...] = heightfield_node_ids(nodes)
    if len(ids) == 0:
        return None
    index: int
    if current_id in ids:
        index = ids.index(current_id)
    else:
        index = 0
    return ids[(index + delta) % len(ids)]


def clamp01(value: float) -> float:
    """clamp to [0, 1]"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def sample_heightfield_normal(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    u: float,
    v: float,
) -> Vector3:
    """hf-local unit normal via finite differences (same form as tile/gpu)"""
    size_x: float = heightfield.size_x.value
    size_z: float = heightfield.size_z.value
    du: float = 1.0 / size_x if size_x > 1e-8 else 0.01
    dv: float = 1.0 / size_z if size_z > 1e-8 else 0.01
    u0: float = clamp01(u - du)
    u1: float = clamp01(u + du)
    v0: float = clamp01(v - dv)
    v1: float = clamp01(v + dv)
    h_l: float = sample_height(catalog, heightfield, u0, v)
    h_r: float = sample_height(catalog, heightfield, u1, v)
    h_d: float = sample_height(catalog, heightfield, u, v0)
    h_u: float = sample_height(catalog, heightfield, u, v1)
    span_x: float = max(size_x * (u1 - u0), 1e-6)
    span_z: float = max(size_z * (v1 - v0), 1e-6)
    nx: float = -(h_r - h_l) / span_x
    ny: float = 1.0
    nz: float = -(h_u - h_d) / span_z
    length: float = sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return Vector3(0.0, 1.0, 0.0)
    return Vector3(nx / length, ny / length, nz / length)


def project_heightfield_apply(
    slots: list[GeneratedSlot],
    params: ProjectHeightfieldParams,
    context: BuildContext,
) -> list[GeneratedSlot]:
    """keep world xz; set world y from terrain + relative generator y lift"""
    catalog: HeightmapCatalog | None = context.heightmap_catalog
    if catalog is None:
        return slots
    hf_node: Node | None = resolve_heightfield_node(
        context.nodes, params.heightfield_node_id.value
    )
    if hf_node is None or hf_node.heightfield is None:
        return slots
    heightfield: Heightfield = hf_node.heightfield
    offset: float = params.offset.value
    align: bool = params.align_to_normal.value
    group_world: Matrix = world_matrix(context.nodes, context.group_id)
    inv_group: Matrix = matrix_invert(group_world)
    hf_world: Matrix = world_matrix(context.nodes, hf_node.id)
    inv_hf: Matrix = matrix_invert(hf_world)
    group_rot: Quaternion = world_rotation(context.nodes, context.group_id)
    inv_group_rot: Quaternion = quaternion_invert(group_rot)
    size_x: float = heightfield.size_x.value
    size_z: float = heightfield.size_z.value
    min_y: float = 0.0
    if len(slots) > 0:
        min_y = min(slot.transform.translation.y for slot in slots)
    result: list[GeneratedSlot] = []
    slot: GeneratedSlot
    for slot in slots:
        slot_world_matrix: Matrix = matrix_multiply(
            transform_matrix(slot.transform), group_world
        )
        world_pos: Vector3 = matrix_translation(slot_world_matrix)
        hf_local: Vector3 = vector3_transform(world_pos, inv_hf)
        u: float
        v: float
        u, v = world_xz_to_uv(hf_local.x, hf_local.z, size_x, size_z)
        height: float = sample_height(catalog, heightfield, u, v)
        surface_hf: Vector3 = Vector3(hf_local.x, height + offset, hf_local.z)
        surface_world: Vector3 = vector3_transform(surface_hf, hf_world)
        lift: float = float(slot.transform.translation.y) - min_y
        desired_world: Vector3 = Vector3(
            world_pos.x, surface_world.y + lift, world_pos.z
        )
        local_pos: Vector3 = vector3_transform(desired_world, inv_group)
        local_rotation: Quaternion = slot.transform.rotation
        if align:
            local_normal: Vector3 = sample_heightfield_normal(
                catalog, heightfield, u, v
            )
            world_normal: Vector3 = vector3_normalize(
                transform_direction(hf_world, local_normal)
            )
            world_forward: Vector3 = transform_direction(
                slot_world_matrix, Vector3(0.0, 0.0, 1.0)
            )
            aligned_world: Quaternion = quaternion_align_y(
                world_normal, world_forward
            )
            local_rotation = quaternion_multiply(inv_group_rot, aligned_world)
        result.append(
            GeneratedSlot(
                Transform(
                    local_pos,
                    local_rotation,
                    slot.transform.scale,
                ),
                slot.role,
            )
        )
    return result


project_heightfield_spec: ModifierSpec = ModifierSpec(
    kind="project_heightfield",
    label="Project heightfield",
    params_type=ProjectHeightfieldParams,
    apply=project_heightfield_apply,
)
