"""rasterize road-sections polylines into absolute ribbon + coverage buffers"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import hypot

from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    RoadSectionsHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
)
from builder.heightfield.paint import touch_heightmap_buffer
from builder.heightfield.road_sections.overlay import shadow_centerline_for_polyline
from builder.heightfield.road_sections.part_catalog import RoadPartsCatalog
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
    road_sections_ribbon_radius_m,
)
from builder.heightfield.vector.coords import ordered_child_nodes
from builder.heightfield.vector.overlay import overlay_step_metres, surface_y
from builder.heightfield.vector.rasterize import (
    find_heightfield_group_for_descendant,
)
from builder.heightfield.vector.stamp import (
    apply_road_absolute,
    clip_texel_range,
    ribbon_plateau_weight,
    world_aabb_to_texel_range,
)
from builder.heightfield.vector.vector_types import (
    RibbonScalarAtXz,
    VectorApplyMode,
    VectorRasterContext,
)
from builder.scene.scene_types import Node
from builder.scene.visibility import node_effectively_visible


def clear_buffer_samples_fast(buffer: HeightmapBuffer) -> None:
    """zero every sample (slice assign is faster than a python loop at 4k²)"""
    count: int = len(buffer.samples)
    if count == 0:
        return
    buffer.samples[:] = [0.0] * count


def sparse_centerline_xz(
    points: Sequence[tuple[float, float]],
    step_metres: float,
) -> list[tuple[float, float]]:
    """densify polyline xz to ~step_metres along arc length"""
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]]
    spacing: float = max(1e-6, step_metres)
    knots: list[tuple[float, float]] = []
    ax: float
    az: float
    ax, az = points[0]
    knots.append((ax, az))
    traveled: float = 0.0
    next_at: float = spacing
    index: int
    for index in range(len(points) - 1):
        bx: float
        bz: float
        bx, bz = points[index + 1]
        dx: float = bx - ax
        dz: float = bz - az
        length: float = hypot(dx, dz)
        if length < 1e-8:
            ax, az = bx, bz
            continue
        inv_len: float = 1.0 / length
        while next_at <= traveled + length + 1e-9:
            t: float = (next_at - traveled) * inv_len
            if t > 1.0:
                t = 1.0
            knots.append((ax + dx * t, az + dz * t))
            next_at += spacing
        traveled += length
        ax, az = bx, bz
    end_x: float
    end_z: float
    end_x, end_z = points[-1]
    last: tuple[float, float] = knots[-1]
    if hypot(end_x - last[0], end_z - last[1]) > 1e-6:
        knots.append((end_x, end_z))
    return knots


def sparse_grade_knots(
    points: Sequence[tuple[float, float]],
    step_metres: float,
    surface_at_xz: RibbonScalarAtXz,
) -> list[tuple[float, float, float]]:
    """sample absolute surface height sparsely along arc length"""
    knots_xz: list[tuple[float, float]] = sparse_centerline_xz(
        points, step_metres
    )
    return [
        (x, z, float(surface_at_xz(x, z))) for x, z in knots_xz
    ]


# grade sampling step relative to paint texel size (metres)
_grade_texel_scale: float = 2.0


def grade_step_metres(
    size_x: float, size_z: float, width: int, height: int
) -> float:
    """sparse underlay sample spacing for ribbon grade (~2 texels, ≥ overlay)"""
    texel_x: float = size_x / float(max(width, 1))
    texel_z: float = size_z / float(max(height, 1))
    texel: float = min(texel_x, texel_z)
    return max(overlay_step_metres, texel * _grade_texel_scale)


def distance_and_t_on_segment(
    x: float,
    z: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> tuple[float, float]:
    """distance to segment and clamped param t in [0, 1]"""
    dx: float = bx - ax
    dz: float = bz - az
    length_sq: float = dx * dx + dz * dz
    t: float = 0.0
    if length_sq > 1e-12:
        t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / length_sq))
    px: float = ax + dx * t
    pz: float = az + dz * t
    return hypot(x - px, z - pz), t


def clear_buffer_region(
    buffer: HeightmapBuffer,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> None:
    """zero samples in an inclusive-exclusive texel rectangle"""
    width: int = buffer.width
    y: int
    for y in range(max(0, y0), min(buffer.height, y1)):
        row: int = y * width
        start: int = row + max(0, x0)
        end: int = row + min(width, x1)
        if start < end:
            buffer.samples[start:end] = [0.0] * (end - start)


def polyline_world_aabb(
    points: Sequence[tuple[float, float]],
    pad_metres: float,
) -> tuple[float, float, float, float] | None:
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


def stamp_grade_segment(
    context: VectorRasterContext,
    ax: float,
    az: float,
    ay: float,
    bx: float,
    bz: float,
    by: float,
) -> None:
    """sweep one grade segment aabb; height lerps ay→by along the foot"""
    if hypot(bx - ax, bz - az) < 1e-8:
        return
    pad: float = context.radius_metres
    ranges: tuple[int, int, int, int] | None = world_aabb_to_texel_range(
        min(ax, bx) - pad,
        max(ax, bx) + pad,
        min(az, bz) - pad,
        max(az, bz) + pad,
        context.size_x,
        context.size_z,
        context.width,
        context.height,
    )
    if ranges is None:
        return
    clipped: tuple[int, int, int, int] | None = clip_texel_range(
        ranges[0], ranges[1], ranges[2], ranges[3], context
    )
    if clipped is None:
        return
    half_width: float = context.radius_metres
    x0: int
    x1: int
    y0: int
    y1: int
    x0, x1, y0, y1 = clipped
    inv_w: float = 1.0 / float(context.width)
    inv_h: float = 1.0 / float(context.height)
    y: int
    for y in range(y0, y1):
        wz: float = ((float(y) + 0.5) * inv_h - 0.5) * context.size_z
        x: int
        for x in range(x0, x1):
            wx: float = ((float(x) + 0.5) * inv_w - 0.5) * context.size_x
            dist: float
            t: float
            dist, t = distance_and_t_on_segment(wx, wz, ax, az, bx, bz)
            weight: float = ribbon_plateau_weight(dist, half_width)
            if weight <= 0.0:
                continue
            height: float = ay + (by - ay) * t
            apply_road_absolute(
                context,
                y * context.width + x,
                height,
                weight,
            )


def stamp_road_centerline_ribbon(
    context: VectorRasterContext,
    points: Sequence[tuple[float, float]],
    surface_at_xz: RibbonScalarAtXz,
    *,
    grade_step_m: float | None = None,
) -> None:
    """path-local aabb sweep; absolute y from sparse underlay grade knots"""
    if len(points) < 2:
        return
    step: float = (
        float(grade_step_m)
        if grade_step_m is not None
        else grade_step_metres(
            context.size_x, context.size_z, context.width, context.height
        )
    )
    knots: list[tuple[float, float, float]] = sparse_grade_knots(
        points, step, surface_at_xz
    )
    if len(knots) < 2:
        return
    index: int
    for index in range(len(knots) - 1):
        ax: float
        az: float
        ay: float
        bx: float
        bz: float
        by: float
        ax, az, ay = knots[index]
        bx, bz, by = knots[index + 1]
        stamp_grade_segment(context, ax, az, ay, bx, bz, by)


def layer_has_ribbon_flatten(
    nodes: dict[str, Node], layer_node_id: str
) -> bool:
    """true if any usage group under the layer has ribbon flatten enabled"""
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_node_id):
        if shp.road_shp is None:
            continue
        usage_node: Node
        for usage_node in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage_node.road_usage_group
            if group is not None and group.ribbon_flatten:
                return True
    return False


def _aabb_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """true if world xz aabbs overlap (min_x, max_x, min_z, max_z)"""
    return not (
        a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]
    )


def rasterize_road_sections_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    layer_node_id: str,
    heightmap_id: HeightmapId,
    coverage_id: HeightmapId,
    *,
    layer_index: int,
    road_parts: RoadPartsCatalog | None = None,
    dirty_world_aabb: tuple[float, float, float, float] | None = None,
    defer_gpu: bool = False,
) -> bool:
    """clear and rebuild road-sections ribbon buffers from flatten-enabled groups

    stamps the fit/placement centerline when road_parts is set; otherwise the
    authored polyline. dirty_world_aabb limits clear + restamp to that band.
    """
    layer_node: Node | None = nodes.get(layer_node_id)
    if layer_node is None or layer_node.road_sections_layer is None:
        return False
    buffer: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if buffer is None:
        return False
    coverage_buffer: HeightmapBuffer | None = catalog.buffers.get(coverage_id)
    if coverage_buffer is None:
        return False
    heightfield_group_id: str | None = find_heightfield_group_for_descendant(
        nodes, layer_node_id
    )
    if heightfield_group_id is None:
        return False
    layout = heightfield_layout(heightfield)
    size_x: float = layout.size_x
    size_z: float = layout.size_z
    touch_dirty: tuple[int, int, int, int] | None = None
    if dirty_world_aabb is None:
        clear_buffer_samples_fast(buffer)
        clear_buffer_samples_fast(coverage_buffer)
    else:
        dirty_range: tuple[int, int, int, int] | None = (
            world_aabb_to_texel_range(
                dirty_world_aabb[0],
                dirty_world_aabb[1],
                dirty_world_aabb[2],
                dirty_world_aabb[3],
                size_x,
                size_z,
                buffer.width,
                buffer.height,
            )
        )
        if dirty_range is not None:
            dx0: int
            dx1: int
            dy0: int
            dy1: int
            dx0, dx1, dy0, dy1 = dirty_range
            touch_dirty = dirty_range
            clear_buffer_region(buffer, dx0, dx1, dy0, dy1)
            clear_buffer_region(coverage_buffer, dx0, dx1, dy0, dy1)
    if (
        not node_effectively_visible(nodes, layer_node_id)
        or not layer_has_ribbon_flatten(nodes, layer_node_id)
    ):
        touch_heightmap_buffer(
            catalog,
            heightmap_id,
            dirty_texels=touch_dirty,
            defer_gpu=defer_gpu,
        )
        touch_heightmap_buffer(
            catalog,
            coverage_id,
            dirty_texels=touch_dirty,
            defer_gpu=defer_gpu,
        )
        return True
    if road_parts is not None:
        road_parts.ensure_loaded()

    def shadow_surface_at_xz(x: float, z: float) -> float:
        # underlay under this layer (same stack the gpu shadow drape uses)
        return surface_y(
            catalog,
            heightfield,
            x,
            z,
            size_x,
            size_z,
            0.0,
            below_layer_index=layer_index,
        )

    grade_step: float = grade_step_metres(
        size_x, size_z, buffer.width, buffer.height
    )
    clip_x0: int | None = None
    clip_x1: int | None = None
    clip_y0: int | None = None
    clip_y1: int | None = None
    if touch_dirty is not None:
        clip_x0, clip_x1, clip_y0, clip_y1 = touch_dirty
    base_context: VectorRasterContext = VectorRasterContext(
        samples=buffer.samples,
        width=buffer.width,
        height=buffer.height,
        size_x=size_x,
        size_z=size_z,
        radius_metres=road_sections_ribbon_radius_m,
        offset=0.0,
        intensity=1.0,
        apply_mode=VectorApplyMode.stroke,
        brush_kind="add",
        sculpt_metres=True,
        flatten_target=None,
        below_at_uv=None,
        ribbon_height_at_xz=None,
        ribbon_half_width_at_xz=None,
        coverage_samples=coverage_buffer.samples,
        clip_x0=clip_x0,
        clip_x1=clip_x1,
        clip_y0=clip_y0,
        clip_y1=clip_y1,
    )
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_node_id):
        if shp.road_shp is None:
            continue
        if not node_effectively_visible(nodes, shp.id):
            continue
        usage_node: Node
        for usage_node in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage_node.road_usage_group
            if group is None or not group.ribbon_flatten:
                continue
            if not node_effectively_visible(nodes, usage_node.id):
                continue
            # stamp radius = full kit width (centerline half-width param)
            ribbon_radius: float = road_sections_ribbon_radius_m
            if road_parts is not None:
                ribbon_radius = road_parts.width_for_kit(group.kit)
            group_context: VectorRasterContext = replace(
                base_context, radius_metres=ribbon_radius
            )
            polyline: RoadPolyline
            for polyline in group.polylines:
                points: list[tuple[float, float]]
                if road_parts is None:
                    points = list(polyline.points)
                else:
                    points = shadow_centerline_for_polyline(
                        group,
                        polyline,
                        road_parts,
                        step_m=grade_step,
                    )
                if dirty_world_aabb is not None:
                    poly_aabb: tuple[float, float, float, float] | None = (
                        polyline_world_aabb(points, ribbon_radius)
                    )
                    if poly_aabb is None or not _aabb_intersects(
                        dirty_world_aabb, poly_aabb
                    ):
                        continue
                stamp_road_centerline_ribbon(
                    group_context,
                    points,
                    shadow_surface_at_xz,
                    grade_step_m=grade_step,
                )
    touch_heightmap_buffer(
        catalog,
        heightmap_id,
        dirty_texels=touch_dirty,
        defer_gpu=defer_gpu,
    )
    touch_heightmap_buffer(
        catalog,
        coverage_id,
        dirty_texels=touch_dirty,
        defer_gpu=defer_gpu,
    )
    return True


def regenerate_road_sections_sources(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    *,
    road_parts: RoadPartsCatalog | None = None,
) -> None:
    """re-rasterize every road-sections edit layer on a heightfield"""
    layer_index: int
    layer: HeightEditLayer
    for layer_index, layer in enumerate(heightfield.edit_layers):
        match layer.height:
            case RoadSectionsHeightSource(
                layer_node_id=layer_node_id,
                heightmap_id=heightmap_id,
                coverage_id=coverage_id,
            ):
                rasterize_road_sections_layer(
                    catalog,
                    heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    coverage_id,
                    layer_index=layer_index,
                    road_parts=road_parts,
                )
            case _:
                pass


def regenerate_road_sections_for_node(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group_id: str,
    layer_node_id: str,
    *,
    road_parts: RoadPartsCatalog | None = None,
    dirty_world_aabb: tuple[float, float, float, float] | None = None,
    defer_gpu: bool = False,
) -> bool:
    """re-rasterize the road-sections source that references layer_node_id"""
    group: Node | None = nodes.get(heightfield_group_id)
    if group is None or group.heightfield is None:
        return False
    layer_index: int
    layer: HeightEditLayer
    for layer_index, layer in enumerate(group.heightfield.edit_layers):
        match layer.height:
            case RoadSectionsHeightSource(
                layer_node_id=source_layer_id,
                heightmap_id=heightmap_id,
                coverage_id=coverage_id,
            ) if source_layer_id == layer_node_id:
                return rasterize_road_sections_layer(
                    catalog,
                    group.heightfield,
                    nodes,
                    layer_node_id,
                    heightmap_id,
                    coverage_id,
                    layer_index=layer_index,
                    road_parts=road_parts,
                    dirty_world_aabb=dirty_world_aabb,
                    defer_gpu=defer_gpu,
                )
            case _:
                pass
    return False


def find_road_sections_layer_ancestor(
    nodes: dict[str, Node], node_id: str
) -> str | None:
    """walk parents to the nearest road sections layer id"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return None
        if node.road_sections_layer is not None:
            return current_id
        current_id = node.parent_id
    return None


# pad beyond ribbon width when building dirty aabb from live edits (metres)
_dirty_edit_slack_m: float = 64.0


def _expand_dirty_from_group(
    group: RoadUsageGroupRecipe,
    *,
    road_parts: RoadPartsCatalog | None,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> tuple[bool, float, float, float, float]:
    """fold one flatten usage group's polylines into a running aabb"""
    if not group.ribbon_flatten:
        return (False, min_x, max_x, min_z, max_z)
    radius: float = road_sections_ribbon_radius_m
    if road_parts is not None:
        radius = road_parts.width_for_kit(group.kit)
    pad: float = radius + _dirty_edit_slack_m
    found: bool = False
    polyline: RoadPolyline
    for polyline in group.polylines:
        box: tuple[float, float, float, float] | None = polyline_world_aabb(
            polyline.points, pad
        )
        if box is None:
            continue
        found = True
        min_x = min(min_x, box[0])
        max_x = max(max_x, box[1])
        min_z = min(min_z, box[2])
        max_z = max(max_z, box[3])
    return (found, min_x, max_x, min_z, max_z)


def vertex_neighborhood_aabb(
    points: Sequence[tuple[float, float]],
    vertex_index: int,
    pad_metres: float,
) -> tuple[float, float, float, float] | None:
    """padded aabb of one vertex and its adjacent open-polyline knots"""
    count: int = len(points)
    if count == 0 or vertex_index < 0 or vertex_index >= count:
        return None
    neighbor: list[tuple[float, float]] = [points[vertex_index]]
    if vertex_index > 0:
        neighbor.append(points[vertex_index - 1])
    if vertex_index + 1 < count:
        neighbor.append(points[vertex_index + 1])
    return polyline_world_aabb(neighbor, pad_metres)


def dirty_aabb_for_road_endpoints(
    group: RoadUsageGroupRecipe,
    endpoints: Sequence[RoadEndpointRef],
    *,
    road_parts: RoadPartsCatalog | None = None,
) -> tuple[float, float, float, float] | None:
    """union neighborhood aabbs for specific road polyline vertices"""
    pad: float = road_sections_ribbon_radius_m
    if road_parts is not None:
        pad = road_parts.width_for_kit(group.kit)
    dirty: tuple[float, float, float, float] | None = None
    endpoint: RoadEndpointRef
    for endpoint in endpoints:
        poly: RoadPolyline | None = polyline_by_id(
            group, endpoint.polyline_id
        )
        if poly is None:
            continue
        local: tuple[float, float, float, float] | None = (
            vertex_neighborhood_aabb(
                poly.points, endpoint.vertex_index, pad
            )
        )
        if local is None:
            continue
        if dirty is None:
            dirty = local
        else:
            dirty = (
                min(dirty[0], local[0]),
                max(dirty[1], local[1]),
                min(dirty[2], local[2]),
                max(dirty[3], local[3]),
            )
    return dirty


def dirty_aabb_for_nodes(
    nodes: dict[str, Node],
    node_ids: Sequence[str],
    *,
    road_parts: RoadPartsCatalog | None = None,
) -> tuple[float, float, float, float] | None:
    """union padded aabb of flatten polylines under the given nodes"""
    min_x: float = 1e30
    max_x: float = -1e30
    min_z: float = 1e30
    max_z: float = -1e30
    found: bool = False
    node_id: str
    for node_id in node_ids:
        node: Node | None = nodes.get(node_id)
        if node is None:
            continue
        if node.road_usage_group is not None:
            hit: bool
            hit, min_x, max_x, min_z, max_z = _expand_dirty_from_group(
                node.road_usage_group,
                road_parts=road_parts,
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
            )
            found = found or hit
            continue
        layer_id: str | None = find_road_sections_layer_ancestor(
            nodes, node_id
        )
        if layer_id is None:
            continue
        shp: Node
        for shp in ordered_child_nodes(nodes, layer_id):
            if shp.road_shp is None:
                continue
            usage: Node
            for usage in ordered_child_nodes(nodes, shp.id):
                if usage.road_usage_group is None:
                    continue
                hit, min_x, max_x, min_z, max_z = _expand_dirty_from_group(
                    usage.road_usage_group,
                    road_parts=road_parts,
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                )
                found = found or hit
    if not found:
        return None
    return (min_x, max_x, min_z, max_z)


def regenerate_road_sections_touching(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    node_ids: list[str],
    *,
    road_parts: RoadPartsCatalog | None = None,
    dirty_world_aabb: tuple[float, float, float, float] | None = None,
    defer_gpu: bool = True,
) -> None:
    """re-rasterize road-sections layers affected by edited nodes"""
    touched_layers: set[str] = set()
    node_id: str
    for node_id in node_ids:
        layer_id: str | None = find_road_sections_layer_ancestor(
            nodes, node_id
        )
        if layer_id is not None:
            touched_layers.add(layer_id)
    dirty: tuple[float, float, float, float] | None = dirty_world_aabb
    if dirty is None:
        dirty = dirty_aabb_for_nodes(
            nodes, node_ids, road_parts=road_parts
        )
    layer_id_touched: str
    for layer_id_touched in touched_layers:
        hf_id: str | None = find_heightfield_group_for_descendant(
            nodes, layer_id_touched
        )
        if hf_id is None:
            continue
        regenerate_road_sections_for_node(
            catalog,
            nodes,
            hf_id,
            layer_id_touched,
            road_parts=road_parts,
            dirty_world_aabb=dirty,
            defer_gpu=defer_gpu,
        )
