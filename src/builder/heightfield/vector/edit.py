"""create and edit vector layers / shapes under a heightfield"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pyray import Vector3

from builder.generators.param_types import FloatParam
from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    HeightLayerSource,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog, HeightmapId
from builder.heightfield.paint import create_paint_heightmap, paint_resolution_for_heightfield
from builder.heightfield.vector.registry import (
    default_vector_shape_params,
    get_vector_shape_spec,
)
from builder.heightfield.vector.vector_types import (
    VectorLayerRecipe,
    VectorPointRecipe,
    VectorShapeRecipe,
)
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node, NodeRole, transform_at


def default_shape_recipe(kind: str) -> VectorShapeRecipe:
    """build a default VectorShapeRecipe for a registered kind"""
    spec = get_vector_shape_spec(kind)
    return VectorShapeRecipe(
        kind=kind,
        apply_mode=spec.default_apply_mode,
        brush_kind="add",
        offset=FloatParam(0.5, label="Offset", step=0.25),
        intensity=FloatParam(1.0, label="Intensity", step=0.05),
        radius=FloatParam(
            4.0, label="Radius", step=0.25, minimum=0.1
        ),
        params=default_vector_shape_params(kind),
        enabled=True,
    )


def default_vector_point_recipe() -> VectorPointRecipe:
    """default per-point recipe (width scale 1)"""
    return VectorPointRecipe(
        width_scale=FloatParam(
            1.0, label="Width scale", step=0.05, minimum=0.05
        )
    )


def make_vector_point(
    parent_id: str, name: str, x: float, z: float
) -> Node:
    """one editable vector_point under a shape"""
    return Node(
        id=new_node_id(),
        parent_id=parent_id,
        transform=transform_at(Vector3(x, 0.0, z)),
        mesh_id=None,
        vector_point=default_vector_point_recipe(),
        name=name,
        role=NodeRole.vector_point,
    )


def default_polyline_nodes(shape_id: str) -> list[Node]:
    """three-point open polyline defaults"""
    return [
        make_vector_point(shape_id, "Point 0", -16.0, 0.0),
        make_vector_point(shape_id, "Point 1", 0.0, 8.0),
        make_vector_point(shape_id, "Point 2", 16.0, 0.0),
    ]


def default_circle_nodes(shape_id: str) -> list[Node]:
    """center point for a circle"""
    return [make_vector_point(shape_id, "Center", 0.0, 0.0)]


def default_rectangle_nodes(shape_id: str) -> list[Node]:
    """two opposite corners for a rectangle"""
    return [
        make_vector_point(shape_id, "Corner 0", -12.0, -8.0),
        make_vector_point(shape_id, "Corner 1", 12.0, 8.0),
    ]


def default_points_for_kind(kind: str, shape_id: str) -> list[Node]:
    """default geometry points for a shape kind"""
    match kind:
        case "polyline":
            return default_polyline_nodes(shape_id)
        case "circle":
            return default_circle_nodes(shape_id)
        case "rectangle":
            return default_rectangle_nodes(shape_id)
        case _:
            return default_polyline_nodes(shape_id)


def append_vector_layer_nodes(
    heightfield: Heightfield,
    heightfield_group_id: str,
    source: HeightLayerSource,
    *,
    label: str,
    default_shape_kind: str,
) -> tuple[Heightfield, list[Node]]:
    """shared VectorLayer + shape + points; source already bound to layer id"""
    layer_id: str
    match source:
        case VectorHeightSource(layer_node_id=layer_node_id):
            layer_id = layer_node_id
        case VectorSculptHeightSource(layer_node_id=layer_node_id):
            layer_id = layer_node_id
        case VectorRoadHeightSource(layer_node_id=layer_node_id):
            layer_id = layer_node_id
        case _:
            raise TypeError("vector append requires a vector height source")
    shape_id: str = new_node_id()
    shape_recipe: VectorShapeRecipe = default_shape_recipe(default_shape_kind)
    layer_node: Node = Node(
        id=layer_id,
        parent_id=heightfield_group_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        vector_layer=VectorLayerRecipe(),
        name=label,
    )
    shape_node: Node = Node(
        id=shape_id,
        parent_id=layer_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        vector_shape=shape_recipe,
        name=get_vector_shape_spec(default_shape_kind).label,
    )
    point_nodes: list[Node] = default_points_for_kind(
        default_shape_kind, shape_id
    )
    edit: HeightEditLayer = HeightEditLayer(name=label, height=source)
    updated: Heightfield = replace(
        heightfield,
        edit_layers=heightfield.edit_layers + (edit,),
    )
    return (updated, [layer_node, shape_node, *point_nodes])


def append_vector_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    heightfield_group_id: str,
    *,
    cache_dir: Path,
    label: str = "Vector",
    default_shape_kind: str = "polyline",
) -> tuple[Heightfield, list[Node]]:
    """create VectorLayer node + default shape + VectorHeightSource edit layer"""
    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    heightmap_id: HeightmapId = create_paint_heightmap(
        catalog,
        label=f"{label} Raster",
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
    layer_id: str = new_node_id()
    return append_vector_layer_nodes(
        heightfield,
        heightfield_group_id,
        VectorHeightSource(layer_node_id=layer_id, heightmap_id=heightmap_id),
        label=label,
        default_shape_kind=default_shape_kind,
    )


def append_vector_sculpt_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    heightfield_group_id: str,
    *,
    cache_dir: Path,
    label: str = "Vector Sculpt",
    default_shape_kind: str = "polyline",
) -> tuple[Heightfield, list[Node]]:
    """create VectorLayer + default shape + VectorSculptHeightSource edit layer"""
    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    heightmap_id: HeightmapId = create_paint_heightmap(
        catalog,
        label=f"{label} Raster",
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
    layer_id: str = new_node_id()
    return append_vector_layer_nodes(
        heightfield,
        heightfield_group_id,
        VectorSculptHeightSource(
            layer_node_id=layer_id, heightmap_id=heightmap_id
        ),
        label=label,
        default_shape_kind=default_shape_kind,
    )


def default_road_shape_recipe(kind: str) -> VectorShapeRecipe:
    """default shape recipe for a road layer (width + intensity; height from points)"""
    spec = get_vector_shape_spec(kind)
    return VectorShapeRecipe(
        kind=kind,
        apply_mode=spec.default_apply_mode,
        brush_kind="add",
        offset=FloatParam(0.0, label="Offset", step=0.25),
        intensity=FloatParam(1.0, label="Intensity", step=0.05),
        radius=FloatParam(
            8.0, label="Width", step=0.5, minimum=0.1
        ),
        params=default_vector_shape_params(kind),
        enabled=True,
    )


def append_vector_road_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    heightfield_group_id: str,
    *,
    cache_dir: Path,
    label: str = "Road",
    default_shape_kind: str = "polyline",
) -> tuple[Heightfield, list[Node]]:
    """create VectorLayer + default shape + VectorRoadHeightSource edit layer"""
    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    heightmap_id: HeightmapId = create_paint_heightmap(
        catalog,
        label=f"{label} Raster",
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
    layer_id: str = new_node_id()
    shape_id: str = new_node_id()
    shape_recipe: VectorShapeRecipe = default_road_shape_recipe(
        default_shape_kind
    )
    layer_node: Node = Node(
        id=layer_id,
        parent_id=heightfield_group_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        vector_layer=VectorLayerRecipe(),
        name=label,
    )
    shape_node: Node = Node(
        id=shape_id,
        parent_id=layer_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        vector_shape=shape_recipe,
        name=get_vector_shape_spec(default_shape_kind).label,
    )
    point_nodes: list[Node] = default_points_for_kind(
        default_shape_kind, shape_id
    )
    edit: HeightEditLayer = HeightEditLayer(
        name=label,
        height=VectorRoadHeightSource(
            layer_node_id=layer_id, heightmap_id=heightmap_id
        ),
    )
    updated: Heightfield = replace(
        heightfield,
        edit_layers=heightfield.edit_layers + (edit,),
    )
    return (updated, [layer_node, shape_node, *point_nodes])


def add_vector_shape(
    layer_node_id: str,
    kind: str,
    *,
    label: str | None = None,
    road: bool = False,
) -> list[Node]:
    """create a shape node and default points under an existing vector layer"""
    shape_id: str = new_node_id()
    recipe: VectorShapeRecipe = (
        default_road_shape_recipe(kind)
        if road
        else default_shape_recipe(kind)
    )
    name: str = label if label is not None else get_vector_shape_spec(kind).label
    shape_node: Node = Node(
        id=shape_id,
        parent_id=layer_node_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        vector_shape=recipe,
        name=name,
    )
    return [shape_node, *default_points_for_kind(kind, shape_id)]


def append_polyline_point(nodes: dict[str, Node], shape_id: str) -> Node:
    """append one vector_point after the last, continuing the last segment"""
    existing: list[Node] = [
        node
        for node in nodes.values()
        if node.parent_id == shape_id and node.role is NodeRole.vector_point
    ]
    count: int = len(existing)
    x: float
    z: float
    if count == 0:
        x = 0.0
        z = 0.0
    elif count == 1:
        last = existing[-1].transform.translation
        x = float(last.x) + 16.0
        z = float(last.z)
    else:
        prev = existing[-2].transform.translation
        last = existing[-1].transform.translation
        x = float(last.x) + (float(last.x) - float(prev.x))
        z = float(last.z) + (float(last.z) - float(prev.z))
    return make_vector_point(shape_id, f"Point {count}", x, z)


def delete_vector_points(
    nodes: dict[str, Node], point_ids: list[str]
) -> list[str]:
    """return point ids that are safe to delete (role vector_point only)"""
    removable: list[str] = []
    point_id: str
    for point_id in point_ids:
        node: Node | None = nodes.get(point_id)
        if node is None:
            continue
        if node.role is NodeRole.vector_point:
            removable.append(point_id)
    return removable


def strip_vector_sources_from_heightfield(
    heightfield: Heightfield,
    layer_node_ids: set[str],
) -> Heightfield:
    """drop edit layers whose vector source points at a removed layer node"""
    if not layer_node_ids:
        return heightfield
    kept: list[HeightEditLayer] = []
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        match layer.height:
            case VectorHeightSource(layer_node_id=layer_node_id) if (
                layer_node_id in layer_node_ids
            ):
                continue
            case VectorSculptHeightSource(layer_node_id=layer_node_id) if (
                layer_node_id in layer_node_ids
            ):
                continue
            case VectorRoadHeightSource(layer_node_id=layer_node_id) if (
                layer_node_id in layer_node_ids
            ):
                continue
            case _:
                kept.append(layer)
    if len(kept) == len(heightfield.edit_layers):
        return heightfield
    return replace(heightfield, edit_layers=tuple(kept))
