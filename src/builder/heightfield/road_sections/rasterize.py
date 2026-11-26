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
    RoadPolyline,
    RoadUsageGroupRecipe,
    road_sections_ribbon_radius_m,
)
from builder.heightfield.vector.coords import ordered_child_nodes
from builder.heightfield.vector.overlay import overlay_step_metres, surface_y
from builder.heightfield.vector.rasterize import (
    find_heightfield_group_for_descendant,
)
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    path_stroke_weight,
    world_aabb_to_texel_range,
)
from builder.heightfield.vector.vector_types import (
    RibbonScalarAtXz,
    RibbonTargetAtXz,
    VectorApplyMode,
    VectorRasterContext,
)
from builder.scene.scene_types import Node


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


def distance_to_segment(
    x: float,
    z: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> float:
    """distance from xz to one open segment"""
    dx: float = bx - ax
    dz: float = bz - az
    length_sq: float = dx * dx + dz * dz
    t: float = 0.0
    if length_sq > 1e-12:
        t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / length_sq))
    px: float = ax + dx * t
    pz: float = az + dz * t
    return hypot(x - px, z - pz)


def stamp_segment_ribbon(
    context: VectorRasterContext,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> None:
    """stamp one segment with a local aabb (avoids whole-polyline bbox cost)"""
    pad: float = context.radius_metres
    min_x: float = min(ax, bx) - pad
    max_x: float = max(ax, bx) + pad
    min_z: float = min(az, bz) - pad
    max_z: float = max(az, bz) + pad
    ranges: tuple[int, int, int, int] | None = world_aabb_to_texel_range(
        min_x,
        max_x,
        min_z,
        max_z,
        context.size_x,
        context.size_z,
        context.width,
        context.height,
    )
    if ranges is None:
        return
    x0: int
    x1: int
    y0: int
    y1: int
    x0, x1, y0, y1 = ranges
    inv_w: float = 1.0 / float(context.width)
    inv_h: float = 1.0 / float(context.height)
    y: int
    for y in range(y0, y1):
        v: float = (float(y) + 0.5) * inv_h
        wz: float = (v - 0.5) * context.size_z
        x: int
        for x in range(x0, x1):
            u: float = (float(x) + 0.5) * inv_w
            wx: float = (u - 0.5) * context.size_x
            dist: float = distance_to_segment(wx, wz, ax, az, bx, bz)
            weight: float = path_stroke_weight(context, dist, x=wx, z=wz)
            if weight <= 0.0:
                continue
            apply_weighted_sample(
                context,
                y * context.width + x,
                u,
                v,
                weight,
            )


def segment_foot_height_fn(
    ax: float,
    az: float,
    bx: float,
    bz: float,
    surface_at_xz: RibbonScalarAtXz,
) -> RibbonTargetAtXz:
    """absolute height = surface under the centerline foot (flat cross-section)"""
    dx: float = bx - ax
    dz: float = bz - az
    length_sq: float = dx * dx + dz * dz

    def ribbon_height_at(x: float, z: float, _below: float) -> float:
        t: float = 0.0
        if length_sq > 1e-12:
            t = max(
                0.0,
                min(1.0, ((x - ax) * dx + (z - az) * dz) / length_sq),
            )
        return float(surface_at_xz(ax + dx * t, az + dz * t))

    return ribbon_height_at


def stamp_road_centerline_ribbon(
    context: VectorRasterContext,
    points: Sequence[tuple[float, float]],
    surface_at_xz: RibbonScalarAtXz,
) -> None:
    """stamp ribbon; height = same surface_y the shadow uses, at centerline foot"""
    if len(points) < 2:
        return
    # match shadow draping densify (overlay_step_metres)
    stroke: list[tuple[float, float]] = sparse_centerline_xz(
        points,
        overlay_step_metres,
    )
    if len(stroke) < 2:
        return
    index: int
    for index in range(len(stroke) - 1):
        ax: float
        az: float
        bx: float
        bz: float
        ax, az = stroke[index]
        bx, bz = stroke[index + 1]
        if hypot(bx - ax, bz - az) < 1e-8:
            continue
        stamped: VectorRasterContext = VectorRasterContext(
            samples=context.samples,
            width=context.width,
            height=context.height,
            size_x=context.size_x,
            size_z=context.size_z,
            radius_metres=context.radius_metres,
            offset=context.offset,
            intensity=context.intensity,
            apply_mode=context.apply_mode,
            brush_kind=context.brush_kind,
            sculpt_metres=True,
            flatten_target=None,
            below_at_uv=context.below_at_uv,
            ribbon_height_at_xz=segment_foot_height_fn(
                ax, az, bx, bz, surface_at_xz
            ),
            ribbon_half_width_at_xz=None,
            coverage_samples=context.coverage_samples,
        )
        stamp_segment_ribbon(stamped, ax, az, bx, bz)


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
) -> bool:
    """clear and rebuild road-sections ribbon buffers from flatten-enabled groups

    stamps the fit/placement centerline when road_parts is set; otherwise the
    authored polyline.
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
    clear_buffer_samples_fast(buffer)
    clear_buffer_samples_fast(coverage_buffer)
    if not layer_has_ribbon_flatten(nodes, layer_node_id):
        touch_heightmap_buffer(catalog, heightmap_id)
        touch_heightmap_buffer(catalog, coverage_id)
        return True
    if road_parts is not None:
        road_parts.ensure_loaded()
    size_x: float = layout.size_x
    size_z: float = layout.size_z

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

    def below_at_uv(u: float, v: float) -> float:
        return shadow_surface_at_xz(
            (u - 0.5) * size_x,
            (v - 0.5) * size_z,
        )

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
        below_at_uv=below_at_uv,
        ribbon_height_at_xz=None,
        ribbon_half_width_at_xz=None,
        coverage_samples=coverage_buffer.samples,
    )
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_node_id):
        if shp.road_shp is None:
            continue
        usage_node: Node
        for usage_node in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage_node.road_usage_group
            if group is None or not group.ribbon_flatten:
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
                    # same centerline densify as shadow overlay (default 2 m)
                    points = shadow_centerline_for_polyline(
                        group,
                        polyline,
                        road_parts,
                        step_m=overlay_step_metres,
                    )
                stamp_road_centerline_ribbon(
                    group_context, points, shadow_surface_at_xz
                )
    touch_heightmap_buffer(catalog, heightmap_id)
    touch_heightmap_buffer(catalog, coverage_id)
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


def regenerate_road_sections_touching(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    node_ids: list[str],
    *,
    road_parts: RoadPartsCatalog | None = None,
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
        )
