"""create road sections layers and import shapefile path trees"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pyray import Vector3

from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    RoadSectionsHeightSource,
    heightfield_layout,
    map_world_xz_to_hf_local,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPolyline,
    RoadSectionsLayerRecipe,
    RoadShpRecipe,
    RoadUsageGroupRecipe,
)
from builder.heightfield.road_sections.shp_import import (
    ImportedRoadPart,
    read_road_shp,
)
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node, transform_at


def append_road_sections_edit_layer(
    heightfield: Heightfield,
    heightfield_group_id: str,
    *,
    label: str = "Road Sections",
) -> tuple[Heightfield, list[Node]]:
    """create an empty road sections layer under a heightfield"""
    layer_id: str = new_node_id()
    layer_node: Node = Node(
        id=layer_id,
        parent_id=heightfield_group_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        road_sections_layer=RoadSectionsLayerRecipe(),
        name=label,
    )
    edit: HeightEditLayer = HeightEditLayer(
        name=label,
        height=RoadSectionsHeightSource(layer_node_id=layer_id),
    )
    updated: Heightfield = replace(
        heightfield,
        edit_layers=heightfield.edit_layers + (edit,),
    )
    return (updated, [layer_node])


def remap_part_to_hf_local(
    part: ImportedRoadPart,
    *,
    origin_x: float,
    origin_z: float,
    size_x: float,
    size_z: float,
) -> ImportedRoadPart:
    """convert one part's map-space points into hf-local xz"""
    points: list[tuple[float, float]] = [
        map_world_xz_to_hf_local(
            x,
            z,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        for x, z in part.points
    ]
    return ImportedRoadPart(usage=part.usage, points=points)


def usage_groups_from_parts(
    parts: list[ImportedRoadPart],
) -> list[tuple[str, tuple[RoadPolyline, ...]]]:
    """group imported parts into (usage, polylines) in first-seen order"""
    order: list[str] = []
    buckets: dict[str, list[RoadPolyline]] = {}
    part: ImportedRoadPart
    for part in parts:
        if part.usage not in buckets:
            order.append(part.usage)
            buckets[part.usage] = []
        buckets[part.usage].append(RoadPolyline(points=tuple(part.points)))
    usage: str
    return [(usage, tuple(buckets[usage])) for usage in order]


def import_road_shp(
    layer_id: str,
    shp_path: Path,
    heightfield: Heightfield,
) -> list[Node]:
    """import one shapefile; returns shp node plus one usage group per class

    shapefile xy is map-space (asc origin), mirrored east-west like asc decode
    """
    result = read_road_shp(shp_path)
    layout = heightfield_layout(heightfield)
    origin_x: float = float(heightfield.origin_x.value)
    origin_z: float = float(heightfield.origin_z.value)
    remapped: list[ImportedRoadPart] = [
        remap_part_to_hf_local(
            part,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=layout.size_x,
            size_z=layout.size_z,
        )
        for part in result.parts
    ]
    shp_id: str = new_node_id()
    nodes: list[Node] = [
        Node(
            id=shp_id,
            parent_id=layer_id,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_shp=RoadShpRecipe(source_path=str(shp_path.resolve())),
            name=shp_path.stem,
        )
    ]
    usage: str
    polylines: tuple[RoadPolyline, ...]
    for usage, polylines in usage_groups_from_parts(remapped):
        nodes.append(
            Node(
                id=new_node_id(),
                parent_id=shp_id,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                road_usage_group=RoadUsageGroupRecipe(
                    usage=usage, polylines=polylines
                ),
                name=usage,
            )
        )
    return nodes


def strip_road_sections_sources_from_heightfield(
    heightfield: Heightfield,
    layer_node_ids: set[str],
) -> Heightfield:
    """drop edit layers whose road sections source points at a removed layer"""
    if not layer_node_ids:
        return heightfield
    kept: list[HeightEditLayer] = []
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        match layer.height:
            case RoadSectionsHeightSource(layer_node_id=layer_node_id) if (
                layer_node_id in layer_node_ids
            ):
                continue
            case _:
                kept.append(layer)
    if len(kept) == len(heightfield.edit_layers):
        return heightfield
    return replace(heightfield, edit_layers=tuple(kept))
