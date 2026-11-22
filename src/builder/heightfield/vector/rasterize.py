"""rasterize live vector layer shapes into a derived paint or sculpt buffer"""

from __future__ import annotations

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
from builder.heightfield.sample import sample_height_below
from builder.heightfield.vector.coords import (
    ordered_child_nodes,
    vector_point_xz,
    vector_point_xzyw,
)
from builder.heightfield.vector.polyline import PolylineShapeParams
from builder.heightfield.vector.registry import get_vector_shape_spec
from builder.heightfield.vector.road import (
    absolute_road_points,
    ribbon_half_width_circle,
    ribbon_half_width_polyline,
    ribbon_half_width_rectangle,
    ribbon_height_circle,
    ribbon_height_polyline,
    ribbon_height_rectangle,
    road_half_width_points,
)
from builder.heightfield.vector.stamp import is_flatten_brush
from builder.heightfield.vector.vector_types import (
    RibbonHeightAtXz,
    VectorRasterContext,
    VectorShapeRecipe,
    world_xz_to_uv,
)
from builder.scene.scene_types import Node


def clear_buffer_samples(buffer: HeightmapBuffer) -> None:
    """zero every sample in place"""
    index: int
    for index in range(len(buffer.samples)):
        buffer.samples[index] = 0.0


def road_ribbon_height_fn(
    kind: str,
    points_abs_xzy: list[tuple[float, float, float]],
    closed: bool,
) -> RibbonHeightAtXz:
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


def road_ribbon_half_width_fn(
    kind: str,
    points_xzw: list[tuple[float, float, float]],
    closed: bool,
) -> RibbonHeightAtXz:
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
) -> bool:
    """clear and rebuild one vector layer buffer from its shape children"""
    layer_node: Node | None = nodes.get(layer_node_id)
    if layer_node is None or layer_node.vector_layer is None:
        return False
    buffer: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if buffer is None:
        return False
    heightfield_group_id: str | None = find_heightfield_group_for_descendant(
        nodes, layer_node_id
    )
    if heightfield_group_id is None:
        return False
    layout = heightfield_layout(heightfield)
    clear_buffer_samples(buffer)

    def below_at_uv(u: float, v: float) -> float:
        return sample_height_below(
            catalog, heightfield, layer_index, u, v
        )

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
        spec = get_vector_shape_spec(recipe.kind)
        points: list[tuple[float, float]] = vector_point_xz(
            nodes, child.id, heightfield_group_id
        )
        flatten_target: float | None = None
        ribbon_height: RibbonHeightAtXz | None = None
        ribbon_half_width: RibbonHeightAtXz | None = None
        radius_metres: float = float(recipe.radius.value)
        if road_metres:
            points_xzyw: list[tuple[float, float, float, float]] = (
                vector_point_xzyw(nodes, child.id, heightfield_group_id)
            )
            points_xzy: list[tuple[float, float, float]] = [
                (x, z, y) for x, z, y, _scale in points_xzyw
            ]
            points_abs: list[tuple[float, float, float]] = absolute_road_points(
                points_xzy, surface_at_xz
            )
            points_xzw: list[tuple[float, float, float]] = (
                road_half_width_points(points_xzyw, radius_metres)
            )
            closed: bool = False
            match recipe.params:
                case PolylineShapeParams(closed=closed_param):
                    closed = bool(closed_param.value)
                case _:
                    closed = recipe.kind == "rectangle"
            ribbon_height = road_ribbon_height_fn(
                recipe.kind, points_abs, closed
            )
            ribbon_half_width = road_ribbon_half_width_fn(
                recipe.kind, points_xzw, closed
            )
            if points_xzw:
                radius_metres = max(hw for _x, _z, hw in points_xzw)
        elif sculpt_metres and is_flatten_brush(recipe.brush_kind) and points:
            pu: float
            pv: float
            pu, pv = world_xz_to_uv(
                points[0][0], points[0][1], layout.size_x, layout.size_z
            )
            flatten_target = below_at_uv(pu, pv) + float(
                recipe.offset.value
            )
        context: VectorRasterContext = VectorRasterContext(
            samples=buffer.samples,
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
        )
        spec.rasterize(context, points, recipe.params)
    touch_heightmap_buffer(catalog, heightmap_id)
    return True


def regenerate_vector_sources(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
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
                )
            case VectorRoadHeightSource(
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
                    road_metres=True,
                )
            case _:
                pass


def regenerate_vector_layer_for_node(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group_id: str,
    layer_node_id: str,
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
                )
            case VectorRoadHeightSource(
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
                    road_metres=True,
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


def find_vector_shape_ancestor(
    nodes: dict[str, Node], node_id: str
) -> str | None:
    """walk parents to the nearest VectorShape node id"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return None
        if node.vector_shape is not None:
            return current_id
        current_id = node.parent_id
    return None


def regenerate_vectors_touching(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    node_ids: list[str],
) -> None:
    """re-rasterize vector layers affected by edited nodes"""
    touched_layers: set[str] = set()
    node_id: str
    for node_id in node_ids:
        layer_id: str | None = find_vector_layer_ancestor(nodes, node_id)
        if layer_id is not None:
            touched_layers.add(layer_id)
    layer_id_touched: str
    for layer_id_touched in touched_layers:
        hf_id: str | None = find_heightfield_group_for_descendant(
            nodes, layer_id_touched
        )
        if hf_id is None:
            continue
        regenerate_vector_layer_for_node(
            catalog, nodes, hf_id, layer_id_touched
        )
