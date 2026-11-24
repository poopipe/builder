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

from builder.heightfield.gpu.height_texture import height_texture_key
from builder.heightfield.heightfield_types import (
    Heightfield,
    RoadSectionsHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.road_sections.road_sections_types import (
    RoadPolyline,
    RoadUsageGroupRecipe,
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
from builder.scene.scene_types import Node
from builder.scene.transforms import world_matrix
from builder.view.line_shader import (
    LineBatch,
    LineMeshCache,
    LineShader,
    clear_line_batch,
    draw_split_line_mesh_cache,
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


def usage_overlay_color(usage: str) -> Color:
    """overlay color for a road usage class"""
    return _usage_overlay_colors.get(usage, _default_usage_color)


def road_sections_layer_is_linked(
    heightfield: Heightfield, layer_node_id: str
) -> bool:
    """true if an edit layer references this road sections layer node"""
    edit = None
    for edit in heightfield.edit_layers:
        match edit.height:
            case RoadSectionsHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return True
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
) -> None:
    """queue every polyline for one usage group, colored by usage"""
    color: Color = usage_overlay_color(group.usage)
    polyline: RoadPolyline
    for polyline in group.polylines:
        draw_polyline_flat(
            batch,
            polyline_xz_points(polyline),
            False,
            plane_y,
            thickness_px,
            color,
        )


def draw_usage_group_shadows(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    group: RoadUsageGroupRecipe,
    size_x: float,
    size_z: float,
    thickness_px: float,
) -> None:
    """drape terrain shadows for one usage group's polylines"""
    polyline: RoadPolyline
    for polyline in group.polylines:
        draw_polyline_shadow(
            batch,
            catalog,
            heightfield,
            polyline_xz_points(polyline),
            False,
            size_x,
            size_z,
            thickness_px,
            0.0,
        )


def road_overlay_content_sig(
    nodes: dict[str, Node],
    layer_id: str,
    usage_filter: set[str] | None,
) -> tuple[tuple[str, str, int, int], ...]:
    """stable content fingerprint for drawn usage groups"""
    parts: list[tuple[str, str, int, int]] = []
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
            polyline: RoadPolyline
            for polyline in group.polylines:
                point_count += len(polyline.points)
            parts.append(
                (usage.id, group.usage, len(group.polylines), point_count)
            )
    return tuple(parts)


def road_sections_overlay_cache_key(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    heightfield_group_id: str,
    layer_id: str,
    usage_filter: set[str] | None,
    plane_y: float,
    nodes: dict[str, Node],
) -> tuple[object, ...]:
    """rebuild key: selection, plane, road content, and height when draping"""
    usage_sig: tuple[str, ...] | None = (
        None if usage_filter is None else tuple(sorted(usage_filter))
    )
    height_key: object | None = None
    if usage_filter is not None:
        height_key = height_texture_key(catalog, heightfield)
    return (
        heightfield_group_id,
        layer_id,
        usage_sig,
        float(plane_y),
        height_key,
        road_overlay_content_sig(nodes, layer_id, usage_filter),
    )


def draw_road_sections_overlays(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group: Node,
    line_shader: LineShader,
    line_batch: LineBatch,
    line_mesh_cache: LineMeshCache,
    selected_ids: Sequence[str] = (),
) -> None:
    """draw path overlays for the selected road sections layer"""
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
    cache_key: tuple[object, ...] = road_sections_overlay_cache_key(
        catalog,
        heightfield,
        heightfield_group.id,
        layer.id,
        usage_filter,
        plane_y,
        nodes,
    )
    if line_mesh_cache.key != cache_key:
        shadow_batch: LineBatch = LineBatch()
        flat_batch: LineBatch = LineBatch()
        # terrain shadows only when a usage group is isolated (same as vector shapes)
        if usage_filter is not None:
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
                )
        sync_split_line_mesh_cache(
            line_mesh_cache, cache_key, shadow_batch, flat_batch
        )
        clear_line_batch(line_batch)
    rl_push_matrix()
    rl_mult_matrixf(ffi.cast("float *", matrix_to_float_v(hf_world).v))
    draw_split_line_mesh_cache(line_mesh_cache, line_shader)
    rl_pop_matrix()
