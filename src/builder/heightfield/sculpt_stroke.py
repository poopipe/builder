"""field-uv pick and paint/sculpt-layer stamp for viewport sculpt"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import cos, pi, sin

from pyray import (
    Color,
    Matrix,
    Ray,
    Vector3,
    draw_line_3d,
    matrix_invert,
    vector3_add,
    vector3_transform,
)

from builder.heightfield.brushes.brush_types import BrushStrokeContext
from builder.heightfield.brushes.registry import default_brush_params, get_brush_spec
from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    PaintHeightSource,
    SculptHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog, HeightmapId
from builder.heightfield.paint import stamp_paint_layer
from builder.heightfield.sample import sample_buffer, sample_height
from builder.heightfield.sculpt import stamp_sculpt_layer
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.theme import ui_color_input_focus
from builder.view.draw_cache import DrawCache, fill_world_matrix, sync_draw_cache


class SculptTargetType(IntEnum):
    """which edit-layer source the active sculpt tool writes"""

    paint = 1
    sculpt = 2


@dataclass(frozen=True)
class FieldUvHit:
    """ray hit on a heightfield mapped to field uv"""

    group_id: str
    u: float
    v: float
    world_point: Vector3


@dataclass(frozen=True)
class SculptToolTarget:
    """selected paint or sculpt layer for the sculpt tool"""

    layer: HeightEditLayer
    layer_index: int
    target_type: SculptTargetType
    heightmap_id: HeightmapId


@dataclass
class SculptStrokeState:
    """active LMB sculpt stroke"""

    group_id: str
    layer_index: int
    heightmap_id: HeightmapId
    flatten_target: float | None
    target_type: SculptTargetType


def selected_sculpt_tool_target(
    group: Node, layer_index: int | None
) -> SculptToolTarget | None:
    """return the selected edit layer when it is paint or sculpt"""
    if group.heightfield is None or layer_index is None:
        return None
    if layer_index < 0 or layer_index >= len(group.heightfield.edit_layers):
        return None
    layer: HeightEditLayer = group.heightfield.edit_layers[layer_index]
    match layer.height:
        case PaintHeightSource(heightmap_id=heightmap_id):
            return SculptToolTarget(
                layer=layer,
                layer_index=layer_index,
                target_type=SculptTargetType.paint,
                heightmap_id=heightmap_id,
            )
        case SculptHeightSource(heightmap_id=heightmap_id):
            return SculptToolTarget(
                layer=layer,
                layer_index=layer_index,
                target_type=SculptTargetType.sculpt,
                heightmap_id=heightmap_id,
            )
        case _:
            return None


def selected_paint_layer(
    group: Node, layer_index: int | None
) -> tuple[HeightEditLayer, HeightmapId] | None:
    """return the selected edit layer when it is a paint source"""
    target: SculptToolTarget | None = selected_sculpt_tool_target(group, layer_index)
    if target is None or target.target_type is not SculptTargetType.paint:
        return None
    return (target.layer, target.heightmap_id)


def pick_field_uv(
    scene: Scene,
    catalog: HeightmapCatalog,
    cache: DrawCache,
    ray: Ray,
    group_id: str,
) -> FieldUvHit | None:
    """intersect the selected heightfield's local xz plane; return field uv"""
    node: Node | None = scene.nodes.get(group_id)
    if node is None or node.heightfield is None:
        return None
    heightfield: Heightfield = node.heightfield
    layout = heightfield_layout(heightfield)
    sync_draw_cache(scene, cache)
    world: Matrix = fill_world_matrix(scene, cache, group_id)
    local_from_world: Matrix = matrix_invert(world)
    origin_local: Vector3 = vector3_transform(ray.position, local_from_world)
    # transform direction as a vector (w=0): point at origin+dir minus origin
    dir_point: Vector3 = vector3_transform(
        vector3_add(ray.position, ray.direction), local_from_world
    )
    dir_local: Vector3 = Vector3(
        dir_point.x - origin_local.x,
        dir_point.y - origin_local.y,
        dir_point.z - origin_local.z,
    )
    if abs(dir_local.y) < 1e-8:
        return None
    # intersect local y = 0 plane
    t: float = -origin_local.y / dir_local.y
    if t < 0.0:
        return None
    local_hit: Vector3 = Vector3(
        origin_local.x + dir_local.x * t,
        0.0,
        origin_local.z + dir_local.z * t,
    )
    half_x: float = 0.5 * layout.size_x
    half_z: float = 0.5 * layout.size_z
    if (
        local_hit.x < -half_x
        or local_hit.x > half_x
        or local_hit.z < -half_z
        or local_hit.z > half_z
    ):
        return None
    u: float = (local_hit.x + half_x) / layout.size_x
    v: float = (local_hit.z + half_z) / layout.size_z
    world_point: Vector3 = vector3_transform(local_hit, world)
    # lift world point to composited height for feedback
    height: float = sample_height(catalog, heightfield, u, v)
    world_point = Vector3(world_point.x, world_point.y + height, world_point.z)
    return FieldUvHit(group_id=group_id, u=u, v=v, world_point=world_point)


def world_radius_to_uv(
    heightfield: Heightfield, radius_metres: float
) -> tuple[float, float]:
    """convert world-space brush radius to field uv radii"""
    layout = heightfield_layout(heightfield)
    ru: float = radius_metres / layout.size_x if layout.size_x > 1e-8 else 0.0
    rv: float = radius_metres / layout.size_z if layout.size_z > 1e-8 else 0.0
    return (max(ru, 1e-6), max(rv, 1e-6))


def begin_sculpt_stroke(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    target: SculptToolTarget,
    group_id: str,
    u: float,
    v: float,
) -> SculptStrokeState:
    """capture flatten target at press for paint or sculpt layers"""
    flatten_target: float | None
    match target.target_type:
        case SculptTargetType.paint:
            buffer = catalog.buffers.get(target.heightmap_id)
            paint_at: float = 0.0
            if buffer is not None:
                paint_at = sample_buffer(buffer, u, v, wrap=True)
            flatten_target = paint_at
        case SculptTargetType.sculpt:
            # flatten to the visible composited surface under the cursor
            flatten_target = sample_height(catalog, heightfield, u, v)
    return SculptStrokeState(
        group_id=group_id,
        layer_index=target.layer_index,
        heightmap_id=target.heightmap_id,
        flatten_target=flatten_target,
        target_type=target.target_type,
    )


def apply_sculpt_stamp(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    stroke: SculptStrokeState,
    brush_kind: str,
    u: float,
    v: float,
    radius_metres: float,
    intensity: float,
) -> bool:
    """stamp the active brush at field uv"""
    radius_u: float
    radius_v: float
    radius_u, radius_v = world_radius_to_uv(heightfield, radius_metres)
    get_brush_spec(brush_kind)
    match stroke.target_type:
        case SculptTargetType.paint:
            context: BrushStrokeContext = BrushStrokeContext(
                center_u=u,
                center_v=v,
                radius_u=radius_u,
                radius_v=radius_v,
                intensity=intensity,
                flatten_target=stroke.flatten_target,
            )
            params = default_brush_params(brush_kind)
            return stamp_paint_layer(
                catalog, stroke.heightmap_id, brush_kind, params, context
            )
        case SculptTargetType.sculpt:
            return stamp_sculpt_layer(
                catalog,
                heightfield,
                stroke.layer_index,
                brush_kind,
                u,
                v,
                radius_u,
                radius_v,
                intensity,
                stroke.flatten_target,
            )


sculpt_preview_y_bias: float = 0.15
sculpt_preview_segments: int = 48


def draw_sculpt_brush_preview(
    center: Vector3,
    radius_metres: float,
    *,
    color: Color | None = None,
    segments: int = sculpt_preview_segments,
) -> None:
    """draw a horizontal ring at the brush footprint (slight y bias vs terrain)"""
    if radius_metres <= 0.0 or segments < 3:
        return
    ring_color: Color = ui_color_input_focus if color is None else color
    y: float = center.y + sculpt_preview_y_bias
    i: int
    for i in range(segments):
        a0: float = (2.0 * pi * float(i)) / float(segments)
        a1: float = (2.0 * pi * float(i + 1)) / float(segments)
        p0: Vector3 = Vector3(
            center.x + cos(a0) * radius_metres,
            y,
            center.z + sin(a0) * radius_metres,
        )
        p1: Vector3 = Vector3(
            center.x + cos(a1) * radius_metres,
            y,
            center.z + sin(a1) * radius_metres,
        )
        draw_line_3d(p0, p1, ring_color)
