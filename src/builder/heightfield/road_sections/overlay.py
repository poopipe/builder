"""viewport overlays for road sections polylines"""

from __future__ import annotations

from collections.abc import Sequence

from pyray import (
    Color,
    Matrix,
    ffi,
    matrix_to_float_v,
    rl_mult_matrixf,
    rl_pop_matrix,
    rl_push_matrix,
)

from builder.heightfield.gpu.heightfield_gpu import (
    HeightfieldGpuState,
    line_height_drape_for_overlay,
)
from builder.heightfield.heightfield_types import (
    Heightfield,
    RoadSectionsHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.road_sections.fit import (
    FittedPiece,
    fit_polyline_pieces,
    fitted_centerline_xz,
)
from builder.heightfield.road_sections.part_catalog import (
    RoadKit,
    RoadPartsCatalog,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPiecePlacement,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
)
from builder.heightfield.vector.coords import (
    node_hf_local_position,
    ordered_child_nodes,
)
from builder.heightfield.vector.overlay import draw_polyline_flat, draw_polyline_shadow
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    shadow_line_pixels,
)
from builder.heightfield.vector.work_plane_point import draw_work_plane_point
from builder.scene.scene_types import Node
from builder.scene.transforms import world_matrix
from builder.view.line_shader import (
    LineBatch,
    LineHeightDrape,
    LineMeshCache,
    LineShader,
    clear_line_batch,
    draw_split_line_mesh_cache,
    flush_line_batch,
    line_shadow_depth_bias,
    sync_split_line_mesh_cache,
)

# stable palette by osm/gtt usage; unknown falls back to white-ish
_usage_overlay_colors: dict[str, Color] = {
    "provincial": Color(220, 40, 40, 255),
    "primary": Color(240, 80, 60, 255),
    "trunk": Color(255, 120, 40, 255),
    "secondary": Color(255, 180, 50, 255),
    "tertiary": Color(255, 220, 80, 255),
    "residential": Color(120, 200, 255, 255),
    "living_street": Color(160, 210, 255, 255),
    "unclassified": Color(180, 180, 200, 255),
    "service": Color(140, 140, 160, 255),
    "track": Color(160, 120, 60, 255),
    "trackunclassified": Color(140, 110, 70, 255),
    "softtrack": Color(180, 150, 90, 255),
    "dirtroad": Color(200, 140, 80, 255),
    "footway": Color(80, 220, 140, 255),
    "path": Color(60, 200, 120, 255),
    "pedestrian": Color(100, 240, 180, 255),
    "cycleway": Color(80, 160, 255, 255),
    "bridleway": Color(180, 100, 200, 255),
    "steps": Color(220, 80, 200, 255),
}
_default_usage_color: Color = Color(220, 220, 220, 255)
_selected_line_color: Color = Color(255, 255, 255, 255)
_selected_line_pixels: float = overlay_line_pixels * 1.75
_vertex_cube_size: float = 1.2


def usage_overlay_color(usage: str) -> Color:
    """overlay color for a road usage class"""
    return _usage_overlay_colors.get(usage, _default_usage_color)


def road_sections_layer_index(
    heightfield: Heightfield, layer_node_id: str
) -> int | None:
    """edit-layer index of the RoadSectionsHeightSource for this layer, if any"""
    layer_index: int
    edit = None
    for layer_index, edit in enumerate(heightfield.edit_layers):
        match edit.height:
            case RoadSectionsHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return layer_index
            case _:
                pass
    return None


def road_sections_layer_is_linked(
    heightfield: Heightfield, layer_node_id: str
) -> bool:
    """true if an edit layer references this road sections layer node"""
    return road_sections_layer_index(heightfield, layer_node_id) is not None


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


def selected_road_sections_layer_id(
    nodes: dict[str, Node],
    heightfield_group_id: str,
    selected_ids: Sequence[str],
) -> str | None:
    """road sections layer under this heightfield touched by selection"""
    selected_id: str
    for selected_id in selected_ids:
        layer_id: str | None = find_road_sections_layer_ancestor(
            nodes, selected_id
        )
        if layer_id is None:
            continue
        layer: Node | None = nodes.get(layer_id)
        if layer is None or layer.parent_id != heightfield_group_id:
            continue
        return layer_id
    return None


def road_sections_plane_y(
    nodes: dict[str, Node], layer_id: str, heightfield_group_id: str
) -> float:
    """work-plane height from the road sections layer in hf-local space"""
    return float(
        node_hf_local_position(nodes, layer_id, heightfield_group_id).y
    )


def selected_usage_group_ids(
    nodes: dict[str, Node],
    layer_id: str,
    selected_ids: Sequence[str],
) -> set[str] | None:
    """usage group ids to draw, or none to draw every usage under the layer

    selecting a usage group isolates it; selecting the layer or shp shows all
    """
    usage_ids: set[str] = set()
    selected_id: str
    for selected_id in selected_ids:
        if find_road_sections_layer_ancestor(nodes, selected_id) != layer_id:
            continue
        node: Node | None = nodes.get(selected_id)
        if node is None:
            continue
        if node.road_usage_group is not None:
            usage_ids.add(selected_id)
    if usage_ids:
        return usage_ids
    return None


def polyline_xz_points(polyline: RoadPolyline) -> list[tuple[float, float]]:
    """copy compact polyline points for vector shadow helpers"""
    return [(float(x), float(z)) for x, z in polyline.points]


def draw_usage_group_polylines(
    batch: LineBatch,
    group: RoadUsageGroupRecipe,
    plane_y: float,
    thickness_px: float,
    *,
    selected_polyline_ids: set[str] | None = None,
    omit_polyline_ids: set[str] | None = None,
) -> None:
    """queue every polyline for one usage group, colored by usage"""
    color: Color = usage_overlay_color(group.usage)
    selected: set[str] = selected_polyline_ids or set()
    omit: set[str] = omit_polyline_ids or set()
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id in omit or polyline.id in selected:
            continue
        draw_polyline_flat(
            batch,
            polyline_xz_points(polyline),
            False,
            plane_y,
            thickness_px,
            color,
        )
    for polyline in group.polylines:
        if polyline.id in omit or polyline.id not in selected:
            continue
        draw_polyline_flat(
            batch,
            polyline_xz_points(polyline),
            False,
            plane_y,
            _selected_line_pixels,
            _selected_line_color,
        )


def draw_polyline_flat_selected(
    batch: LineBatch,
    polyline: RoadPolyline,
    plane_y: float,
) -> None:
    """queue one selected polyline on the work plane (live drag path)"""
    draw_polyline_flat(
        batch,
        polyline_xz_points(polyline),
        False,
        plane_y,
        _selected_line_pixels,
        _selected_line_color,
    )


def endpoint_is_selected(
    selected_endpoints: Sequence[RoadEndpointRef],
    polyline_id: str,
    vertex_index: int,
) -> bool:
    """true if this vertex is in the endpoint selection"""
    endpoint: RoadEndpointRef
    for endpoint in selected_endpoints:
        if (
            endpoint.polyline_id == polyline_id
            and endpoint.vertex_index == vertex_index
        ):
            return True
    return False


def draw_selected_polyline_vertices(
    group: RoadUsageGroupRecipe,
    plane_y: float,
    selected_polyline_ids: set[str],
    selected_endpoints: Sequence[RoadEndpointRef],
) -> None:
    """draw work-plane point cubes for every vertex on selected polylines"""
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id not in selected_polyline_ids:
            continue
        vertex_index: int
        for vertex_index, (x, z) in enumerate(polyline.points):
            selected: bool = endpoint_is_selected(
                selected_endpoints, polyline.id, vertex_index
            )
            draw_work_plane_point(
                x, plane_y, z, _vertex_cube_size, selected=selected
            )


def fitted_pieces_for_polyline(
    group: RoadUsageGroupRecipe,
    polyline: RoadPolyline,
    road_parts: RoadPartsCatalog,
) -> list[FittedPiece]:
    """placements when present, otherwise a live fit for shadow preview"""
    placed: list[RoadPiecePlacement] = [
        placement
        for placement in group.placements
        if placement.polyline_id == polyline.id
    ]
    if placed:
        return [
            FittedPiece(
                stem=placement.stem,
                x=placement.x,
                z=placement.z,
                yaw_deg=placement.yaw_deg,
                roll_deg=placement.roll_deg,
            )
            for placement in placed
        ]
    road_parts.ensure_loaded()
    kit: RoadKit | None = road_parts.kits.get(group.kit)
    if kit is None:
        return []
    return fit_polyline_pieces(list(polyline.points), kit)


def shadow_centerline_for_polyline(
    group: RoadUsageGroupRecipe,
    polyline: RoadPolyline,
    road_parts: RoadPartsCatalog,
    *,
    step_m: float = 2.0,
) -> list[tuple[float, float]]:
    """fit centerline xz for draping; falls back to authored polyline"""
    pieces: list[FittedPiece] = fitted_pieces_for_polyline(
        group, polyline, road_parts
    )
    if pieces:
        return fitted_centerline_xz(
            pieces,
            length_for_stem=road_parts.length_for_stem,
            corner_for_stem=road_parts.corner_for_stem,
            step_m=step_m,
        )
    return polyline_xz_points(polyline)


def draw_usage_group_shadows(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    group: RoadUsageGroupRecipe,
    size_x: float,
    size_z: float,
    thickness_px: float,
    *,
    road_parts: RoadPartsCatalog,
    omit_polyline_ids: set[str] | None = None,
    gpu_drape: bool = True,
    below_layer_index: int | None = None,
) -> None:
    """queue fit-centerline xz; y from below-layer gpu composite at draw"""
    color: Color = usage_overlay_color(group.usage)
    omit: set[str] = omit_polyline_ids or set()
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id in omit:
            continue
        points: list[tuple[float, float]] = shadow_centerline_for_polyline(
            group, polyline, road_parts
        )
        if len(points) < 2:
            continue
        draw_polyline_shadow(
            batch,
            catalog,
            heightfield,
            points,
            False,
            size_x,
            size_z,
            thickness_px,
            0.0,
            color,
            below_layer_index=below_layer_index,
            gpu_drape=gpu_drape,
        )


def draw_polyline_shadow_authored(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    polyline: RoadPolyline,
    group: RoadUsageGroupRecipe,
    size_x: float,
    size_z: float,
    *,
    gpu_drape: bool = True,
    below_layer_index: int | None = None,
) -> None:
    """queue authored polyline xz — live vertex-drag path"""
    points: list[tuple[float, float]] = polyline_xz_points(polyline)
    if len(points) < 2:
        return
    draw_polyline_shadow(
        batch,
        catalog,
        heightfield,
        points,
        False,
        size_x,
        size_z,
        shadow_line_pixels,
        0.0,
        usage_overlay_color(group.usage),
        below_layer_index=below_layer_index,
        gpu_drape=gpu_drape,
    )


def live_polylines_for_layer(
    nodes: dict[str, Node],
    layer_id: str,
    live_polyline_ids: set[str],
    usage_filter: set[str] | None,
) -> list[tuple[RoadUsageGroupRecipe, RoadPolyline]]:
    """resolve live polyline ids under a road sections layer"""
    found: list[tuple[RoadUsageGroupRecipe, RoadPolyline]] = []
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_id):
        if shp.road_shp is None:
            continue
        usage: Node
        for usage in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage.road_usage_group
            if group is None:
                continue
            if usage_filter is not None and usage.id not in usage_filter:
                continue
            live_id: str
            for live_id in live_polyline_ids:
                polyline: RoadPolyline | None = polyline_by_id(group, live_id)
                if polyline is None:
                    continue
                found.append((group, polyline))
    return found


def road_overlay_content_sig(
    nodes: dict[str, Node],
    layer_id: str,
    usage_filter: set[str] | None,
    *,
    omit_polyline_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[str, str, int, int, int, int, tuple[str, ...]], ...]:
    """stable content fingerprint for drawn usage groups

    omit_polyline_ids: skip those points in the geometry hash so a live vertex
    drag does not rebuild the whole line mesh every frame
    """
    parts: list[tuple[str, str, int, int, int, int, tuple[str, ...]]] = []
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_id):
        if shp.road_shp is None:
            continue
        usage: Node
        for usage in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage.road_usage_group
            if group is None:
                continue
            if usage_filter is not None and usage.id not in usage_filter:
                continue
            point_count: int = 0
            polyline_ids: list[str] = []
            geometry: list[tuple[str, tuple[tuple[float, float], ...]]] = []
            polyline: RoadPolyline
            for polyline in group.polylines:
                point_count += len(polyline.points)
                polyline_ids.append(polyline.id)
                if polyline.id in omit_polyline_ids:
                    continue
                geometry.append((polyline.id, polyline.points))
            # hash so in-place vertex moves invalidate the line mesh cache
            geometry_hash: int = hash(tuple(geometry))
            parts.append(
                (
                    usage.id,
                    group.usage,
                    len(group.polylines),
                    point_count,
                    len(group.placements),
                    geometry_hash,
                    tuple(polyline_ids),
                )
            )
    return tuple(parts)


def road_sections_overlay_cache_key(
    heightfield_group_id: str,
    layer_id: str,
    usage_filter: set[str] | None,
    plane_y: float,
    nodes: dict[str, Node],
    selected_polyline_ids: frozenset[str] = frozenset(),
    *,
    omit_polyline_ids: frozenset[str] = frozenset(),
    skip_shadows: bool = False,
) -> tuple[object, ...]:
    """rebuild key: selection, plane, and road content (y from live heightMap)"""
    usage_sig: tuple[str, ...] | None = (
        None if usage_filter is None else tuple(sorted(usage_filter))
    )
    return (
        heightfield_group_id,
        layer_id,
        usage_sig,
        float(plane_y),
        road_overlay_content_sig(
            nodes,
            layer_id,
            usage_filter,
            omit_polyline_ids=omit_polyline_ids,
        ),
        tuple(sorted(selected_polyline_ids)),
        tuple(sorted(omit_polyline_ids)),
        skip_shadows,
    )


def draw_road_sections_overlays(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group: Node,
    line_shader: LineShader,
    line_batch: LineBatch,
    line_mesh_cache: LineMeshCache,
    selected_ids: Sequence[str] = (),
    selected_polyline_ids: Sequence[str] = (),
    selected_endpoints: Sequence[RoadEndpointRef] = (),
    *,
    road_parts: RoadPartsCatalog | None = None,
    live_polyline_ids: Sequence[str] = (),
    skip_shadows: bool = False,
    heightfield_gpu: HeightfieldGpuState | None = None,
) -> None:
    """draw path overlays for the selected road sections layer

    live_polyline_ids: omit from the mesh cache and draw flat + authored
    shadow each frame (vertex drag)
    """
    heightfield: Heightfield | None = heightfield_group.heightfield
    if heightfield is None:
        return
    selected_layer_id: str | None = selected_road_sections_layer_id(
        nodes, heightfield_group.id, selected_ids
    )
    if selected_layer_id is None:
        return
    layer: Node | None = nodes.get(selected_layer_id)
    if (
        layer is None
        or layer.road_sections_layer is None
        or not road_sections_layer_is_linked(heightfield, layer.id)
    ):
        return
    plane_y: float = road_sections_plane_y(
        nodes, layer.id, heightfield_group.id
    )
    usage_filter: set[str] | None = selected_usage_group_ids(
        nodes, layer.id, selected_ids
    )
    layout = heightfield_layout(heightfield)
    size_x: float = layout.size_x
    size_z: float = layout.size_z
    hf_world: Matrix = world_matrix(nodes, heightfield_group.id)
    selected_poly_set: set[str] = set(selected_polyline_ids)
    live_poly_set: set[str] = set(live_polyline_ids)
    live_omit: frozenset[str] = frozenset(live_poly_set)
    sections_index: int | None = road_sections_layer_index(
        heightfield, layer.id
    )
    use_gpu_drape: bool = heightfield_gpu is not None
    shadow_drape: LineHeightDrape | None = None
    if (
        use_gpu_drape
        and heightfield_gpu is not None
        and not skip_shadows
        and usage_filter is not None
    ):
        shadow_drape = line_height_drape_for_overlay(
            heightfield_gpu,
            catalog,
            heightfield_group.id,
            heightfield,
            sections_index,
        )
    cache_key: tuple[object, ...] = road_sections_overlay_cache_key(
        heightfield_group.id,
        layer.id,
        usage_filter,
        plane_y,
        nodes,
        frozenset(selected_poly_set),
        omit_polyline_ids=live_omit,
        skip_shadows=skip_shadows,
    )
    if line_mesh_cache.key != cache_key:
        shadow_batch: LineBatch = LineBatch()
        flat_batch: LineBatch = LineBatch()
        # terrain shadows only when a usage group is isolated (same as vector shapes)
        if (
            not skip_shadows
            and usage_filter is not None
            and road_parts is not None
        ):
            shp_shadow: Node
            for shp_shadow in ordered_child_nodes(nodes, layer.id):
                if shp_shadow.road_shp is None:
                    continue
                usage_shadow: Node
                for usage_shadow in ordered_child_nodes(nodes, shp_shadow.id):
                    if usage_shadow.road_usage_group is None:
                        continue
                    if usage_shadow.id not in usage_filter:
                        continue
                    draw_usage_group_shadows(
                        shadow_batch,
                        catalog,
                        heightfield,
                        usage_shadow.road_usage_group,
                        size_x,
                        size_z,
                        shadow_line_pixels,
                        road_parts=road_parts,
                        omit_polyline_ids=live_poly_set,
                        gpu_drape=use_gpu_drape,
                        below_layer_index=sections_index,
                    )
        shp: Node
        for shp in ordered_child_nodes(nodes, layer.id):
            if shp.road_shp is None:
                continue
            usage: Node
            for usage in ordered_child_nodes(nodes, shp.id):
                if usage.road_usage_group is None:
                    continue
                if usage_filter is not None and usage.id not in usage_filter:
                    continue
                draw_usage_group_polylines(
                    flat_batch,
                    usage.road_usage_group,
                    plane_y,
                    overlay_line_pixels,
                    selected_polyline_ids=selected_poly_set,
                    omit_polyline_ids=live_poly_set,
                )
        sync_split_line_mesh_cache(
            line_mesh_cache, cache_key, shadow_batch, flat_batch
        )
        clear_line_batch(line_batch)
    rl_push_matrix()
    rl_mult_matrixf(ffi.cast("float *", matrix_to_float_v(hf_world).v))
    draw_split_line_mesh_cache(
        line_mesh_cache, line_shader, shadow_drape=shadow_drape
    )
    if live_poly_set:
        live_pairs: list[tuple[RoadUsageGroupRecipe, RoadPolyline]] = (
            live_polylines_for_layer(
                nodes, layer.id, live_poly_set, usage_filter
            )
        )
        clear_line_batch(line_batch)
        group_live: RoadUsageGroupRecipe
        live_line: RoadPolyline
        for group_live, live_line in live_pairs:
            draw_polyline_shadow_authored(
                line_batch,
                catalog,
                heightfield,
                live_line,
                group_live,
                size_x,
                size_z,
                gpu_drape=use_gpu_drape,
                below_layer_index=sections_index,
            )
        flush_line_batch(
            line_batch,
            line_shader,
            depth_bias=line_shadow_depth_bias,
            height_drape=shadow_drape,
        )
        clear_line_batch(line_batch)
        for group_live, live_line in live_pairs:
            draw_polyline_flat_selected(line_batch, live_line, plane_y)
        flush_line_batch(line_batch, line_shader)
    if selected_poly_set:
        shp_ends: Node
        for shp_ends in ordered_child_nodes(nodes, layer.id):
            if shp_ends.road_shp is None:
                continue
            usage_ends: Node
            for usage_ends in ordered_child_nodes(nodes, shp_ends.id):
                group_ends: RoadUsageGroupRecipe | None = (
                    usage_ends.road_usage_group
                )
                if group_ends is None:
                    continue
                if usage_filter is not None and usage_ends.id not in usage_filter:
                    continue
                draw_selected_polyline_vertices(
                    group_ends,
                    plane_y,
                    selected_poly_set,
                    selected_endpoints,
                )
    rl_pop_matrix()
