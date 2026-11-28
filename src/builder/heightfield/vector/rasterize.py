"""rasterize live vector layer shapes into a derived paint or sculpt buffer"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from builder.generators.param_types import BoolParam, FloatParam
from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
)
from builder.heightfield.paint import touch_heightmap_buffer
from builder.heightfield.road_sections.part_catalog import RoadPartsCatalog
from builder.heightfield.sample import sample_height_below
from builder.heightfield.vector.circle import CircleShapeParams
from builder.heightfield.vector.coords import (
    find_vector_shape_ancestor,
    ordered_child_nodes,
    vector_point_entries,
    vector_point_xz,
    vector_point_xzyw,
)
from builder.heightfield.vector.polyline import PolylineShapeParams
from builder.heightfield.vector.rectangle import (
    RectangleShapeParams,
    oriented_rectangle_hf_corners,
    oriented_rectangle_hf_xzy,
)
from builder.heightfield.vector.registry import get_vector_shape_spec
from builder.heightfield.vector.road import (
    absolute_road_points,
    blend_ribbon_height,
    ribbon_half_width_circle,
    ribbon_half_width_polyline,
    ribbon_half_width_rectangle,
    ribbon_height_circle,
    ribbon_height_polyline,
    ribbon_height_rectangle,
    road_half_width_points,
    road_offset_points,
)
from builder.heightfield.vector.stamp import (
    is_flatten_brush,
    vector_flatten_target,
    world_aabb_to_texel_range,
)
from builder.heightfield.vector.vector_types import (
    BelowAtUv,
    RibbonScalarAtXz,
    RibbonTargetAtXz,
    VectorApplyMode,
    VectorRasterContext,
    VectorShapeRecipe,
    world_xz_to_uv,
)
from builder.path.points import (
    densify_path_positions,
    densify_path_xzyw,
    group_local_position,
    handle_local,
    ordered_path_points,
    path_segments_near_point_ids,
)
from builder.scene.scene_types import Node, NodeRole
from builder.scene.visibility import node_effectively_visible
from pyray import Vector3

if TYPE_CHECKING:
    from builder.heightfield.gpu.sculpt_stamp import GpuSculptShape

WorldAabb = tuple[float, float, float, float]


def stamp_densify_step_metres(spacing: float) -> float:
    """polyline stamp sample spacing matched to heightfield mesh spacing"""
    return max(float(spacing), 1e-3)


def cpu_stamp_sculpt_shape(
    buffer: HeightmapBuffer,
    shape: "GpuSculptShape",
    *,
    size_x: float,
    size_z: float,
    clip_x0: int,
    clip_x1: int,
    clip_y0: int,
    clip_y1: int,
    below_at_uv: BelowAtUv | None,
) -> None:
    """cpu-rasterize one sculpt shape into an already-cleared dirty rect"""
    spec = get_vector_shape_spec(shape.kind)
    params: object
    match shape.kind:
        case "polyline":
            params = PolylineShapeParams(
                closed=BoolParam(shape.closed, label="Closed")
            )
        case "circle":
            params = CircleShapeParams(
                radius=FloatParam(shape.circle_radius, label="Radius")
            )
        case "rectangle":
            params = RectangleShapeParams()
        case _:
            params = spec.params_type()
    flatten_target: float | None = None
    if is_flatten_brush(shape.brush_kind):
        flatten_target = shape.flatten_target
    context: VectorRasterContext = VectorRasterContext(
        raster=buffer,
        width=buffer.width,
        height=buffer.height,
        size_x=size_x,
        size_z=size_z,
        radius_metres=shape.radius_metres,
        offset=shape.offset,
        intensity=shape.intensity,
        apply_mode=shape.apply_mode,
        brush_kind=shape.brush_kind,
        sculpt_metres=True,
        flatten_target=flatten_target,
        below_at_uv=below_at_uv,
        clip_x0=clip_x0,
        clip_x1=clip_x1,
        clip_y0=clip_y0,
        clip_y1=clip_y1,
    )
    spec.rasterize(context, shape.points, params)


def clear_buffer_samples(buffer: HeightmapBuffer) -> None:
    """drop every allocated tile"""
    from builder.heightfield.sparse_height import clear_raster

    clear_raster(buffer)


def clear_buffer_region(
    buffer: HeightmapBuffer,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> None:
    """zero samples in an inclusive-exclusive texel rectangle"""
    from builder.heightfield.sparse_height import clear_raster_region

    clear_raster_region(buffer, x0, x1, y0, y1)


def aabb_intersects(a: WorldAabb, b: WorldAabb) -> bool:
    """true if world xz aabbs overlap (min_x, max_x, min_z, max_z)"""
    return not (
        a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]
    )


def union_aabb(
    a: WorldAabb | None, b: WorldAabb | None
) -> WorldAabb | None:
    """axis-aligned union, or none if both empty"""
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def points_world_aabb(
    points: Sequence[tuple[float, float]],
    pad_metres: float,
) -> WorldAabb | None:
    """padded world xz aabb, or none if empty"""
    if len(points) == 0:
        return None
    xs: list[float] = [p[0] for p in points]
    zs: list[float] = [p[1] for p in points]
    pad: float = max(0.0, pad_metres)
    return (
        min(xs) - pad,
        max(xs) + pad,
        min(zs) - pad,
        max(zs) + pad,
    )


def shape_stamp_aabb_points(
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
) -> tuple[list[tuple[float, float]], float] | None:
    """conservative hf-local corners for stamp aabb (no densify)"""
    recipe: VectorShapeRecipe | None = shape.vector_shape
    if recipe is None or not recipe.enabled:
        return None
    pad: float = float(recipe.radius.value)
    points: list[tuple[float, float]]
    if recipe.kind == "polyline":
        path_points: list[Node] = ordered_path_points(nodes, shape.id)
        if len(path_points) == 0:
            return None
        corners: list[tuple[float, float]] = []
        point: Node
        for point in path_points:
            position: Vector3 = group_local_position(
                nodes, heightfield_group_id, point.id
            )
            corners.append((float(position.x), float(position.z)))
            handle_out: Vector3 = handle_local(
                nodes,
                heightfield_group_id,
                point,
                NodeRole.path_handle_out,
                position,
            )
            handle_in: Vector3 = handle_local(
                nodes,
                heightfield_group_id,
                point,
                NodeRole.path_handle_in,
                position,
            )
            corners.append((float(handle_out.x), float(handle_out.z)))
            corners.append((float(handle_in.x), float(handle_in.z)))
        points = corners
    elif recipe.kind == "rectangle":
        oriented: list[tuple[float, float]] | None = (
            oriented_rectangle_hf_corners(
                nodes, shape.id, heightfield_group_id
            )
        )
        points = oriented if oriented is not None else []
    elif recipe.kind == "circle":
        center: list[tuple[float, float]] = vector_point_xz(
            nodes, shape.id, heightfield_group_id
        )
        if not center:
            return None
        geom_r: float = 8.0
        match recipe.params:
            case CircleShapeParams(radius=radius_param):
                geom_r = max(0.1, float(radius_param.value))
            case _:
                pass
        cx: float
        cz: float
        cx, cz = center[0]
        points = [
            (cx - geom_r, cz),
            (cx + geom_r, cz),
            (cx, cz - geom_r),
            (cx, cz + geom_r),
        ]
    else:
        points = vector_point_xz(nodes, shape.id, heightfield_group_id)
    return points, pad


def shape_world_aabb(
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
) -> WorldAabb | None:
    """padded stamp footprint of one vector shape in hf-local xz"""
    stamped: tuple[list[tuple[float, float]], float] | None = (
        shape_stamp_aabb_points(nodes, shape, heightfield_group_id)
    )
    if stamped is None:
        return None
    points: list[tuple[float, float]]
    pad: float
    points, pad = stamped
    return points_world_aabb(points, pad)


def shape_allows_vertex_local_dirty(shape: Node) -> bool:
    """true for open stroke polylines where a point move is local"""
    recipe: VectorShapeRecipe | None = shape.vector_shape
    if recipe is None or recipe.kind != "polyline":
        return False
    if recipe.apply_mode is not VectorApplyMode.stroke:
        return False
    closed: bool = False
    match recipe.params:
        case PolylineShapeParams(closed=closed_param):
            closed = bool(closed_param.value)
        case _:
            closed = False
    return not closed


def point_neighborhood_aabb(
    nodes: dict[str, Node],
    point_id: str,
    heightfield_group_id: str,
) -> WorldAabb | None:
    """padded aabb of a path point and its adjacent polyline knots"""
    shape_id: str | None = find_vector_shape_ancestor(nodes, point_id)
    if shape_id is None:
        return None
    shape: Node | None = nodes.get(shape_id)
    if shape is None or shape.vector_shape is None:
        return None
    if not shape_allows_vertex_local_dirty(shape):
        return shape_world_aabb(nodes, shape, heightfield_group_id)
    entries: list[tuple[str, float, float]] = vector_point_entries(
        nodes, shape_id, heightfield_group_id
    )
    if not entries:
        return None
    index: int = -1
    entry_i: int
    for entry_i, (entry_id, _x, _z) in enumerate(entries):
        if entry_id == point_id:
            index = entry_i
            break
    if index < 0:
        return None
    neighbor_points: list[tuple[float, float]] = [
        (entries[index][1], entries[index][2])
    ]
    if index > 0:
        neighbor_points.append((entries[index - 1][1], entries[index - 1][2]))
    if index + 1 < len(entries):
        neighbor_points.append((entries[index + 1][1], entries[index + 1][2]))
    pad: float = float(shape.vector_shape.radius.value)
    return points_world_aabb(neighbor_points, pad)


def local_edit_shape_ids(
    nodes: dict[str, Node], node_ids: Sequence[str]
) -> set[str]:
    """stroke-polyline shapes edited only via path-point moves"""
    local_ids: set[str] = set()
    node_id: str
    for node_id in node_ids:
        node: Node | None = nodes.get(node_id)
        if node is None:
            continue
        if node.vector_shape is not None:
            continue
        if node.role is not NodeRole.path_point:
            continue
        shape_id: str | None = find_vector_shape_ancestor(nodes, node_id)
        if shape_id is None:
            continue
        shape: Node | None = nodes.get(shape_id)
        if shape is None or not shape_allows_vertex_local_dirty(shape):
            continue
        local_ids.add(shape_id)
    return local_ids


def local_edit_point_ids(
    nodes: dict[str, Node], node_ids: Sequence[str]
) -> set[str]:
    """path-point ids that drive vertex-local dirty restamps"""
    point_ids: set[str] = set()
    node_id: str
    for node_id in node_ids:
        node: Node | None = nodes.get(node_id)
        if node is None or node.role is not NodeRole.path_point:
            continue
        shape_id: str | None = find_vector_shape_ancestor(nodes, node_id)
        if shape_id is None:
            continue
        shape: Node | None = nodes.get(shape_id)
        if shape is None or not shape_allows_vertex_local_dirty(shape):
            continue
        point_ids.add(node_id)
    return point_ids


def dirty_aabb_for_vector_nodes(
    nodes: dict[str, Node],
    node_ids: Sequence[str],
) -> WorldAabb | None:
    """union stamp aabb of shapes touched by node_ids (current geometry)

    path-point edits on open stroke polylines use adjacent-segment neighborhoods
    """
    dirty: WorldAabb | None = None
    seen_full_shapes: set[str] = set()
    seen_points: set[str] = set()
    node_id: str
    for node_id in node_ids:
        node: Node | None = nodes.get(node_id)
        if node is None:
            continue
        if node.vector_shape is not None:
            if node_id in seen_full_shapes:
                continue
            seen_full_shapes.add(node_id)
            hf_id: str | None = find_heightfield_group_for_descendant(
                nodes, node_id
            )
            if hf_id is None:
                continue
            dirty = union_aabb(
                dirty, shape_world_aabb(nodes, node, hf_id)
            )
            continue
        shape_id: str | None = find_vector_shape_ancestor(nodes, node_id)
        if shape_id is None:
            continue
        shape: Node | None = nodes.get(shape_id)
        if shape is None:
            continue
        hf_for_shape: str | None = find_heightfield_group_for_descendant(
            nodes, shape_id
        )
        if hf_for_shape is None:
            continue
        if (
            node.role is NodeRole.path_point
            and shape_allows_vertex_local_dirty(shape)
        ):
            if node_id in seen_points:
                continue
            seen_points.add(node_id)
            dirty = union_aabb(
                dirty,
                point_neighborhood_aabb(nodes, node_id, hf_for_shape),
            )
            continue
        if shape_id in seen_full_shapes:
            continue
        seen_full_shapes.add(shape_id)
        dirty = union_aabb(
            dirty, shape_world_aabb(nodes, shape, hf_for_shape)
        )
    return dirty


def expand_dirty_for_layer(
    nodes: dict[str, Node],
    layer_id: str,
    heightfield_group_id: str,
    seed: WorldAabb,
    *,
    local_shape_ids: set[str] | None = None,
) -> tuple[WorldAabb, set[str]]:
    """collect restamp ids for shapes overlapping seed; dirty stays the seed

    clip restamp keeps additive stamps correct without growing the clear rect
    """
    local_ids: set[str] = set() if local_shape_ids is None else local_shape_ids
    dirty: WorldAabb = seed
    restamp: set[str] = set(local_ids)
    child: Node
    for child in ordered_child_nodes(nodes, layer_id):
        if child.vector_shape is None or child.id in restamp:
            continue
        if not node_effectively_visible(nodes, child.id):
            continue
        aabb: WorldAabb | None = shape_world_aabb(
            nodes, child, heightfield_group_id
        )
        if aabb is None or not aabb_intersects(dirty, aabb):
            continue
        restamp.add(child.id)
    return dirty, restamp


def bake_below_scratch(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layer_index: int,
    width: int,
    height: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[list[float], int, int, int]:
    """sample underlay once per dirty texel; returns (scratch, x0, y0, stride)"""
    stride: int = max(0, x1 - x0)
    rows: int = max(0, y1 - y0)
    scratch: list[float] = [0.0] * (stride * rows)
    if stride == 0 or rows == 0:
        return scratch, x0, y0, stride
    inv_w: float = 1.0 / float(width)
    inv_h: float = 1.0 / float(height)
    y: int
    for y in range(y0, y1):
        v: float = (float(y) + 0.5) * inv_h
        row: int = (y - y0) * stride
        x: int
        for x in range(x0, x1):
            u: float = (float(x) + 0.5) * inv_w
            scratch[row + (x - x0)] = sample_height_below(
                catalog, heightfield, layer_index, u, v
            )
    return scratch, x0, y0, stride


def road_ribbon_height_fn(
    kind: str,
    points_abs_xzy: list[tuple[float, float, float]],
    closed: bool,
) -> RibbonScalarAtXz:
    """build a ribbon height callback for one road shape (absolute point heights)"""

    def height_at(x: float, z: float) -> float:
        match kind:
            case "circle":
                return ribbon_height_circle(x, z, points_abs_xzy)
            case "rectangle":
                return ribbon_height_rectangle(x, z, points_abs_xzy)
            case _:
                return ribbon_height_polyline(
                    x, z, points_abs_xzy, closed
                )

    return height_at


def road_ribbon_offset_fn(
    kind: str,
    points_offset_xzy: list[tuple[float, float, float]],
    closed: bool,
) -> RibbonScalarAtXz:
    """build a path-offset callback (no surface sample)"""

    def offset_at(x: float, z: float) -> float:
        match kind:
            case "circle":
                return ribbon_height_circle(x, z, points_offset_xzy)
            case "rectangle":
                return ribbon_height_rectangle(x, z, points_offset_xzy)
            case _:
                return ribbon_height_polyline(
                    x, z, points_offset_xzy, closed
                )

    return offset_at


def road_ribbon_half_width_fn(
    kind: str,
    points_xzw: list[tuple[float, float, float]],
    closed: bool,
) -> RibbonScalarAtXz:
    """build a half-width callback for one road shape"""

    def half_width_at(x: float, z: float) -> float:
        match kind:
            case "circle":
                return ribbon_half_width_circle(x, z, points_xzw)
            case "rectangle":
                return ribbon_half_width_rectangle(x, z, points_xzw)
            case _:
                return ribbon_half_width_polyline(
                    x, z, points_xzw, closed
                )

    return half_width_at


def collect_gpu_sculpt_shapes(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    layer_node_id: str,
    heightfield_group_id: str,
    size_x: float,
    size_z: float,
    layer_index: int,
    *,
    spacing: float,
    restamp_ids: set[str] | None,
    local_shape_ids: set[str] | None,
    local_point_ids: set[str] | None,
) -> list["GpuSculptShape"]:
    """build gpu stamp descriptors for enabled sculpt shapes on a layer"""
    from builder.heightfield.gpu.sculpt_stamp import (
        GpuSculptShape,
        build_gpu_sculpt_shape,
    )

    densify_step: float = stamp_densify_step_metres(spacing)
    shapes: list[GpuSculptShape] = []
    child: Node
    for child in ordered_child_nodes(nodes, layer_node_id):
        recipe: VectorShapeRecipe | None = child.vector_shape
        if recipe is None or not recipe.enabled:
            continue
        if not node_effectively_visible(nodes, child.id):
            continue
        if restamp_ids is not None and child.id not in restamp_ids:
            continue
        closed: bool = False
        circle_radius: float = 0.0
        match recipe.params:
            case PolylineShapeParams(closed=closed_param):
                closed = bool(closed_param.value)
            case CircleShapeParams(radius=circle_radius_param):
                circle_radius = max(0.1, float(circle_radius_param.value))
                closed = False
            case _:
                closed = recipe.kind == "rectangle"
        radius_metres: float = float(recipe.radius.value)
        points: list[tuple[float, float]]
        if recipe.kind == "polyline":
            only_segments: set[int] | None = None
            if (
                local_shape_ids is not None
                and child.id in local_shape_ids
                and local_point_ids
            ):
                only_segments = path_segments_near_point_ids(
                    nodes,
                    child.id,
                    local_point_ids,
                    closed=closed,
                )
            densified: list[Vector3] = densify_path_positions(
                nodes,
                child.id,
                heightfield_group_id,
                closed=closed,
                step_metres=densify_step,
                only_segments=only_segments,
            )
            points = [(float(p.x), float(p.z)) for p in densified]
        elif recipe.kind == "rectangle":
            oriented: list[tuple[float, float]] | None = (
                oriented_rectangle_hf_corners(
                    nodes, child.id, heightfield_group_id
                )
            )
            points = oriented if oriented is not None else []
        else:
            points = vector_point_xz(
                nodes, child.id, heightfield_group_id
            )
        flatten_target: float = 0.0
        if is_flatten_brush(recipe.brush_kind) and points:
            pu: float
            pv: float
            pu, pv = world_xz_to_uv(
                points[0][0], points[0][1], size_x, size_z
            )
            below_h: float = sample_height_below(
                catalog, heightfield, layer_index, pu, pv
            )
            flatten_target = vector_flatten_target(
                recipe.brush_kind,
                float(recipe.offset.value),
                below_h,
            )
        built: GpuSculptShape | None = build_gpu_sculpt_shape(
            kind=recipe.kind,
            apply_mode=recipe.apply_mode,
            brush_kind=recipe.brush_kind,
            radius_metres=radius_metres,
            offset=float(recipe.offset.value),
            intensity=float(recipe.intensity.value),
            flatten_target=flatten_target,
            points=points,
            closed=closed,
            circle_radius=circle_radius,
        )
        if built is not None:
            shapes.append(built)
    return shapes


def rasterize_vector_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    layer_node_id: str,
    heightmap_id: HeightmapId,
    *,
    sculpt_metres: bool,
    layer_index: int,
    road_metres: bool = False,
    coverage_id: HeightmapId | None = None,
    dirty_world_aabb: WorldAabb | None = None,
    local_shape_ids: set[str] | None = None,
    local_point_ids: set[str] | None = None,
    defer_gpu: bool = False,
    heightfield_gpu: object | None = None,
) -> bool:
    """clear and rebuild one vector layer buffer from its shape children

    dirty_world_aabb limits clear + restamp to overlapping shapes (additive-safe)
    local_shape_ids: vertex-local edits restamped with neighborhood densify
    heightfield_gpu: when set, vector sculpt stamps on the gpu
    """
    layer_node: Node | None = nodes.get(layer_node_id)
    if layer_node is None or layer_node.vector_layer is None:
        return False
    buffer: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if buffer is None:
        return False
    coverage_buffer: HeightmapBuffer | None = None
    if road_metres:
        if coverage_id is None:
            return False
        coverage_buffer = catalog.buffers.get(coverage_id)
        if coverage_buffer is None:
            return False
    heightfield_group_id: str | None = find_heightfield_group_for_descendant(
        nodes, layer_node_id
    )
    if heightfield_group_id is None:
        return False
    layout = heightfield_layout(heightfield)
    restamp_ids: set[str] | None = None
    clear_x0: int = 0
    clear_x1: int = buffer.width
    clear_y0: int = 0
    clear_y1: int = buffer.height
    touch_dirty: tuple[int, int, int, int] | None = None
    use_gpu_sculpt: bool = (
        sculpt_metres
        and not road_metres
        and heightfield_gpu is not None
    )
    if dirty_world_aabb is None:
        if not use_gpu_sculpt:
            clear_buffer_samples(buffer)
        if coverage_buffer is not None:
            clear_buffer_samples(coverage_buffer)
    else:
        expanded: WorldAabb
        expanded, restamp_ids = expand_dirty_for_layer(
            nodes,
            layer_node_id,
            heightfield_group_id,
            dirty_world_aabb,
            local_shape_ids=local_shape_ids,
        )
        dirty_range: tuple[int, int, int, int] | None = (
            world_aabb_to_texel_range(
                expanded[0],
                expanded[1],
                expanded[2],
                expanded[3],
                layout.size_x,
                layout.size_z,
                buffer.width,
                buffer.height,
            )
        )
        if dirty_range is None:
            # nothing on the field — still bump revision
            touch_heightmap_buffer(catalog, heightmap_id)
            if coverage_id is not None:
                touch_heightmap_buffer(catalog, coverage_id)
            return True
        clear_x0, clear_x1, clear_y0, clear_y1 = dirty_range
        touch_dirty = dirty_range
        if not use_gpu_sculpt:
            clear_buffer_region(buffer, clear_x0, clear_x1, clear_y0, clear_y1)
        if coverage_buffer is not None:
            clear_buffer_region(
                coverage_buffer, clear_x0, clear_x1, clear_y0, clear_y1
            )

    if use_gpu_sculpt:
        from builder.heightfield.gpu.composite import ensure_solid_textures
        from builder.heightfield.gpu.heightfield_gpu import (
            HeightfieldGpuState,
            ensure_height_texture_below,
        )
        from builder.heightfield.gpu.sculpt_stamp import (
            GpuSculptShape,
            gpu_shape_fits,
            gpu_stamp_sculpt_shapes,
        )

        assert isinstance(heightfield_gpu, HeightfieldGpuState)
        collected_shapes: list[GpuSculptShape] = collect_gpu_sculpt_shapes(
            catalog,
            heightfield,
            nodes,
            layer_node_id,
            heightfield_group_id,
            layout.size_x,
            layout.size_z,
            layer_index,
            spacing=layout.spacing,
            restamp_ids=restamp_ids,
            local_shape_ids=local_shape_ids,
            local_point_ids=local_point_ids,
        )
        gpu_shapes: list[GpuSculptShape] = []
        oversized_shapes: list[GpuSculptShape] = []
        collected: GpuSculptShape
        for collected in collected_shapes:
            if gpu_shape_fits(collected):
                gpu_shapes.append(collected)
            else:
                oversized_shapes.append(collected)
        needs_flatten: bool = any(
            shape.brush_mode != 0 for shape in collected_shapes
        )
        ensure_solid_textures(heightfield_gpu.composite)
        assert heightfield_gpu.composite.solid_zero is not None
        below_texture = heightfield_gpu.composite.solid_zero.texture
        if needs_flatten:
            below_ht = ensure_height_texture_below(
                heightfield_gpu,
                catalog,
                heightfield_group_id,
                heightfield,
                layer_index,
            )
            below_texture = below_ht.texture
        gpu_stamp_sculpt_shapes(
            heightfield_gpu.composite,
            heightmap_id,
            buffer,
            size_x=layout.size_x,
            size_z=layout.size_z,
            clear_x0=clear_x0,
            clear_x1=clear_x1,
            clear_y0=clear_y0,
            clear_y1=clear_y1,
            shapes=gpu_shapes,
            below_texture=below_texture,
        )
        if len(oversized_shapes) > 0:

            def below_at_uv_gpu_fallback(u: float, v: float) -> float:
                return sample_height_below(
                    catalog, heightfield, layer_index, u, v
                )

            oversized: GpuSculptShape
            for oversized in oversized_shapes:
                cpu_stamp_sculpt_shape(
                    buffer,
                    oversized,
                    size_x=layout.size_x,
                    size_z=layout.size_z,
                    clip_x0=clear_x0,
                    clip_x1=clear_x1,
                    clip_y0=clear_y0,
                    clip_y1=clear_y1,
                    below_at_uv=(
                        below_at_uv_gpu_fallback if needs_flatten else None
                    ),
                )
        touch_heightmap_buffer(
            catalog,
            heightmap_id,
            dirty_texels=touch_dirty,
            defer_gpu=defer_gpu,
        )
        return True

    below_scratch: list[float] | None = None
    scratch_x0: int = clear_x0
    scratch_y0: int = clear_y0
    scratch_stride: int = 0
    needs_flatten_below: bool = False
    if sculpt_metres and not road_metres:
        child_probe: Node
        for child_probe in ordered_child_nodes(nodes, layer_node_id):
            recipe_probe: VectorShapeRecipe | None = (
                child_probe.vector_shape
            )
            if recipe_probe is None or not recipe_probe.enabled:
                continue
            if (
                restamp_ids is not None
                and child_probe.id not in restamp_ids
            ):
                continue
            if is_flatten_brush(recipe_probe.brush_kind):
                needs_flatten_below = True
                break
    if needs_flatten_below:
        below_scratch, scratch_x0, scratch_y0, scratch_stride = (
            bake_below_scratch(
                catalog,
                heightfield,
                layer_index,
                buffer.width,
                buffer.height,
                clear_x0,
                clear_x1,
                clear_y0,
                clear_y1,
            )
        )

    def below_at_uv(u: float, v: float) -> float:
        if below_scratch is None or scratch_stride <= 0:
            return sample_height_below(
                catalog, heightfield, layer_index, u, v
            )
        x: int = int(u * float(buffer.width) - 0.5)
        y: int = int(v * float(buffer.height) - 0.5)
        if x < scratch_x0 or x >= scratch_x0 + scratch_stride:
            return sample_height_below(
                catalog, heightfield, layer_index, u, v
            )
        if y < scratch_y0 or y >= clear_y1:
            return sample_height_below(
                catalog, heightfield, layer_index, u, v
            )
        return below_scratch[
            (y - scratch_y0) * scratch_stride + (x - scratch_x0)
        ]

    def surface_at_xz(x: float, z: float) -> float:
        su: float
        sv: float
        su, sv = world_xz_to_uv(x, z, layout.size_x, layout.size_z)
        return below_at_uv(su, sv)

    child: Node
    for child in ordered_child_nodes(nodes, layer_node_id):
        recipe: VectorShapeRecipe | None = child.vector_shape
        if recipe is None or not recipe.enabled:
            continue
        if not node_effectively_visible(nodes, child.id):
            continue
        if restamp_ids is not None and child.id not in restamp_ids:
            continue
        spec = get_vector_shape_spec(recipe.kind)
        closed: bool = False
        match recipe.params:
            case PolylineShapeParams(closed=closed_param):
                closed = bool(closed_param.value)
            case _:
                closed = recipe.kind == "rectangle"
        radius_metres: float = float(recipe.radius.value)
        densify_step: float = stamp_densify_step_metres(layout.spacing)
        points: list[tuple[float, float]]
        if recipe.kind == "polyline":
            only_segments: set[int] | None = None
            if (
                local_shape_ids is not None
                and child.id in local_shape_ids
                and local_point_ids
            ):
                only_segments = path_segments_near_point_ids(
                    nodes,
                    child.id,
                    local_point_ids,
                    closed=closed,
                )
            densified: list[Vector3] = densify_path_positions(
                nodes,
                child.id,
                heightfield_group_id,
                closed=closed,
                step_metres=densify_step,
                only_segments=only_segments,
            )
            points = [(float(p.x), float(p.z)) for p in densified]
        elif recipe.kind == "rectangle":
            oriented: list[tuple[float, float]] | None = (
                oriented_rectangle_hf_corners(
                    nodes, child.id, heightfield_group_id
                )
            )
            points = oriented if oriented is not None else []
        else:
            points = vector_point_xz(
                nodes, child.id, heightfield_group_id
            )
        flatten_target: float | None = None
        ribbon_height: RibbonTargetAtXz | None = None
        ribbon_half_width: RibbonScalarAtXz | None = None
        if road_metres:
            knot_xzyw: list[tuple[float, float, float, float]]
            if recipe.kind == "rectangle":
                oriented_xzy: list[tuple[float, float, float]] | None = (
                    oriented_rectangle_hf_xzy(
                        nodes, child.id, heightfield_group_id
                    )
                )
                if oriented_xzy is None:
                    knot_xzyw = []
                else:
                    knot_xzyw = [
                        (x, z, y, 1.0) for x, z, y in oriented_xzy
                    ]
            else:
                knot_xzyw = vector_point_xzyw(
                    nodes, child.id, heightfield_group_id
                )
            ribbon_kind: str = (
                "polyline" if recipe.kind == "polyline" else recipe.kind
            )
            # grade: absolute heights from knots only
            knot_xzy: list[tuple[float, float, float]] = [
                (x, z, y) for x, z, y, _scale in knot_xzyw
            ]
            grade_abs: list[tuple[float, float, float]] = absolute_road_points(
                knot_xzy, surface_at_xz
            )
            grade_fn: RibbonScalarAtXz = road_ribbon_height_fn(
                ribbon_kind, grade_abs, closed
            )
            # drape: surface (passed in) + lerped knot offsets
            offset_xzy: list[tuple[float, float, float]] = road_offset_points(
                knot_xzyw
            )
            offset_fn: RibbonScalarAtXz = road_ribbon_offset_fn(
                ribbon_kind, offset_xzy, closed
            )
            grade_blend: float = max(
                0.0, min(1.0, float(recipe.grade.value))
            )

            def ribbon_height_at(x: float, z: float, below: float) -> float:
                if grade_blend >= 1.0 - 1e-6:
                    return grade_fn(x, z)
                drape: float = below + offset_fn(x, z)
                if grade_blend <= 1e-6:
                    return drape
                return blend_ribbon_height(drape, grade_fn(x, z), grade_blend)

            ribbon_height = ribbon_height_at
            # half-width: densify polyline for smoother scale lerp
            width_xzyw: list[tuple[float, float, float, float]]
            width_closed: bool = closed
            if recipe.kind == "polyline":
                width_xzyw = densify_path_xzyw(
                    nodes,
                    child.id,
                    heightfield_group_id,
                    closed=closed,
                    step_metres=densify_step,
                )
                width_closed = False
            else:
                width_xzyw = knot_xzyw
            points_xzw: list[tuple[float, float, float]] = (
                road_half_width_points(width_xzyw, radius_metres)
            )
            ribbon_half_width = road_ribbon_half_width_fn(
                ribbon_kind,
                points_xzw,
                width_closed,
            )
            if points_xzw:
                radius_metres = max(hw for _x, _z, hw in points_xzw)
        elif sculpt_metres and is_flatten_brush(recipe.brush_kind) and points:
            pu: float
            pv: float
            pu, pv = world_xz_to_uv(
                points[0][0], points[0][1], layout.size_x, layout.size_z
            )
            flatten_target = vector_flatten_target(
                recipe.brush_kind,
                float(recipe.offset.value),
                below_at_uv(pu, pv),
            )
        clip_x0: int | None = None
        clip_x1: int | None = None
        clip_y0: int | None = None
        clip_y1: int | None = None
        if dirty_world_aabb is not None:
            clip_x0 = clear_x0
            clip_x1 = clear_x1
            clip_y0 = clear_y0
            clip_y1 = clear_y1
        context: VectorRasterContext = VectorRasterContext(
            raster=buffer,
            width=buffer.width,
            height=buffer.height,
            size_x=layout.size_x,
            size_z=layout.size_z,
            radius_metres=radius_metres,
            offset=float(recipe.offset.value),
            intensity=float(recipe.intensity.value),
            apply_mode=recipe.apply_mode,
            brush_kind=recipe.brush_kind,
            sculpt_metres=sculpt_metres or road_metres,
            flatten_target=flatten_target,
            below_at_uv=below_at_uv if (sculpt_metres or road_metres) else None,
            ribbon_height_at_xz=ribbon_height,
            ribbon_half_width_at_xz=ribbon_half_width,
            coverage=coverage_buffer,
            clip_x0=clip_x0,
            clip_x1=clip_x1,
            clip_y0=clip_y0,
            clip_y1=clip_y1,
        )
        spec.rasterize(context, points, recipe.params)
    touch_heightmap_buffer(
        catalog,
        heightmap_id,
        dirty_texels=touch_dirty,
        defer_gpu=defer_gpu,
    )
    if coverage_id is not None:
        touch_heightmap_buffer(
            catalog,
            coverage_id,
            dirty_texels=touch_dirty,
            defer_gpu=defer_gpu,
        )
    return True


def regenerate_vector_sources(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    *,
    road_parts: RoadPartsCatalog | None = None,
    heightfield_gpu: object | None = None,
) -> None:
    """re-rasterize every vector edit layer on a heightfield"""
    layer_index: int
    layer: HeightEditLayer
    for layer_index, layer in enumerate(heightfield.edit_layers):
        match layer.height:
            case VectorHeightSource(
                layer_node_id=layer_node_id, heightmap_id=heightmap_id
            ):
                rasterize_vector_layer(
                    catalog,
                    heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=False,
                    layer_index=layer_index,
                    heightfield_gpu=heightfield_gpu,
                )
            case VectorSculptHeightSource(
                layer_node_id=layer_node_id, heightmap_id=heightmap_id
            ):
                rasterize_vector_layer(
                    catalog,
                    heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=True,
                    layer_index=layer_index,
                    heightfield_gpu=heightfield_gpu,
                )
            case VectorRoadHeightSource(
                layer_node_id=layer_node_id,
                heightmap_id=heightmap_id,
                coverage_id=coverage_id,
            ):
                rasterize_vector_layer(
                    catalog,
                    heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=True,
                    layer_index=layer_index,
                    road_metres=True,
                    coverage_id=coverage_id,
                    heightfield_gpu=heightfield_gpu,
                )
            case _:
                pass
    from builder.heightfield.road_sections.rasterize import (
        regenerate_road_sections_sources,
    )

    regenerate_road_sections_sources(
        catalog, heightfield, nodes, road_parts=road_parts
    )


def regenerate_vector_layer_for_node(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group_id: str,
    layer_node_id: str,
    *,
    dirty_world_aabb: WorldAabb | None = None,
    local_shape_ids: set[str] | None = None,
    local_point_ids: set[str] | None = None,
    defer_gpu: bool = False,
    heightfield_gpu: object | None = None,
) -> bool:
    """re-rasterize the vector source that references layer_node_id"""
    group: Node | None = nodes.get(heightfield_group_id)
    if group is None or group.heightfield is None:
        return False
    layer_index: int
    layer: HeightEditLayer
    for layer_index, layer in enumerate(group.heightfield.edit_layers):
        match layer.height:
            case VectorHeightSource(
                layer_node_id=source_layer_id, heightmap_id=heightmap_id
            ) if source_layer_id == layer_node_id:
                return rasterize_vector_layer(
                    catalog,
                    group.heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=False,
                    layer_index=layer_index,
                    dirty_world_aabb=dirty_world_aabb,
                    local_shape_ids=local_shape_ids,
                    local_point_ids=local_point_ids,
                    defer_gpu=defer_gpu,
                    heightfield_gpu=heightfield_gpu,
                )
            case VectorSculptHeightSource(
                layer_node_id=source_layer_id, heightmap_id=heightmap_id
            ) if source_layer_id == layer_node_id:
                return rasterize_vector_layer(
                    catalog,
                    group.heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=True,
                    layer_index=layer_index,
                    dirty_world_aabb=dirty_world_aabb,
                    local_shape_ids=local_shape_ids,
                    local_point_ids=local_point_ids,
                    defer_gpu=defer_gpu,
                    heightfield_gpu=heightfield_gpu,
                )
            case VectorRoadHeightSource(
                layer_node_id=source_layer_id,
                heightmap_id=heightmap_id,
                coverage_id=coverage_id,
            ) if source_layer_id == layer_node_id:
                return rasterize_vector_layer(
                    catalog,
                    group.heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    sculpt_metres=True,
                    layer_index=layer_index,
                    road_metres=True,
                    coverage_id=coverage_id,
                    dirty_world_aabb=dirty_world_aabb,
                    local_shape_ids=local_shape_ids,
                    local_point_ids=local_point_ids,
                    defer_gpu=defer_gpu,
                    heightfield_gpu=heightfield_gpu,
                )
            case _:
                pass
    return False


def find_heightfield_group_for_descendant(
    nodes: dict[str, Node], node_id: str
) -> str | None:
    """walk parents to the owning heightfield group id"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return None
        if node.heightfield is not None:
            return current_id
        current_id = node.parent_id
    return None


def find_vector_layer_ancestor(
    nodes: dict[str, Node], node_id: str
) -> str | None:
    """walk parents to the nearest VectorLayer node id"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return None
        if node.vector_layer is not None:
            return current_id
        current_id = node.parent_id
    return None


def regenerate_vectors_touching(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    node_ids: list[str],
    *,
    prior_dirty_aabb: WorldAabb | None = None,
    defer_gpu: bool = True,
    heightfield_gpu: object | None = None,
) -> None:
    """re-rasterize vector layers affected by edited nodes

    prior_dirty_aabb: footprint before the edit (unioned with current)
    defer_gpu: postpone paint revision bump until the next draw flush
    heightfield_gpu: when set, vector sculpt stamps on the gpu
    """
    touched_layers: set[str] = set()
    node_id: str
    for node_id in node_ids:
        layer_id: str | None = find_vector_layer_ancestor(nodes, node_id)
        if layer_id is not None:
            touched_layers.add(layer_id)
    current_dirty: WorldAabb | None = dirty_aabb_for_vector_nodes(
        nodes, node_ids
    )
    dirty: WorldAabb | None = union_aabb(prior_dirty_aabb, current_dirty)
    local_shapes: set[str] = local_edit_shape_ids(nodes, node_ids)
    local_points: set[str] = local_edit_point_ids(nodes, node_ids)
    layer_id_touched: str
    for layer_id_touched in touched_layers:
        hf_id: str | None = find_heightfield_group_for_descendant(
            nodes, layer_id_touched
        )
        if hf_id is None:
            continue
        regenerate_vector_layer_for_node(
            catalog,
            nodes,
            hf_id,
            layer_id_touched,
            dirty_world_aabb=dirty,
            local_shape_ids=local_shapes,
            local_point_ids=local_points,
            defer_gpu=defer_gpu,
            heightfield_gpu=heightfield_gpu,
        )
