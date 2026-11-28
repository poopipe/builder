"""create road sections layers and import shapefile path trees"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import hypot
from pathlib import Path

from pyray import Vector3

from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    RoadSectionsHeightSource,
    heightfield_layout,
    map_world_xz_to_hf_local,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog, HeightmapId
from builder.heightfield.paint import (
    create_paint_heightmap,
    paint_resolution_for_heightfield,
)
from builder.heightfield.road_sections.part_catalog import require_kit_for_usage
from builder.heightfield.road_sections.place import drop_placements
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadSectionsLayerRecipe,
    RoadShpRecipe,
    RoadUsageGroupRecipe,
    polyline_by_id,
    road_endpoint_snap_m,
    road_sections_raster_max,
)
from builder.heightfield.road_sections.shp_import import (
    ImportedRoadPart,
    read_road_shp,
    simplify_polyline,
)
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node, transform_at


def append_road_sections_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    heightfield_group_id: str,
    *,
    cache_dir: Path,
    label: str = "Road Sections",
) -> tuple[Heightfield, list[Node]]:
    """create an empty road sections layer under a heightfield"""
    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    width = min(width, road_sections_raster_max)
    height = min(height, road_sections_raster_max)
    heightmap_id: HeightmapId = create_paint_heightmap(
        catalog,
        label=f"{label} Raster",
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
    coverage_id: HeightmapId = create_paint_heightmap(
        catalog,
        label=f"{label} Coverage",
        width=width,
        height=height,
        cache_dir=cache_dir,
    )
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
        height=RoadSectionsHeightSource(
            layer_node_id=layer_id,
            heightmap_id=heightmap_id,
            coverage_id=coverage_id,
        ),
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
        buckets[part.usage].append(
            RoadPolyline(id=new_node_id(), points=tuple(part.points))
        )
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
                    usage=usage,
                    kit=require_kit_for_usage(usage),
                    polylines=polylines,
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


def polyline_length_m(polyline: RoadPolyline) -> float:
    """xz path length in metres"""
    if len(polyline.points) < 2:
        return 0.0
    total: float = 0.0
    index: int
    for index in range(len(polyline.points) - 1):
        ax: float
        az: float
        ax, az = polyline.points[index]
        bx: float
        bz: float
        bx, bz = polyline.points[index + 1]
        total += hypot(bx - ax, bz - az)
    return total


def polyline_ids_shorter_than(
    group: RoadUsageGroupRecipe,
    max_length_m: float,
) -> list[str]:
    """ids of polylines whose length is strictly less than max_length_m"""
    return [
        polyline.id
        for polyline in group.polylines
        if polyline_length_m(polyline) < max_length_m
    ]


def replace_usage_polylines(
    group: RoadUsageGroupRecipe,
    polylines: Sequence[RoadPolyline],
) -> RoadUsageGroupRecipe:
    """return a usage group with a new polyline tuple"""
    return replace(group, polylines=tuple(polylines))


def delete_polylines(
    group: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
) -> RoadUsageGroupRecipe:
    """drop polylines by id and their placements"""
    drop: set[str] = set(polyline_ids)
    kept: list[RoadPolyline] = [
        polyline for polyline in group.polylines if polyline.id not in drop
    ]
    kept_ids: set[str] = {polyline.id for polyline in kept}
    placements = tuple(
        placement
        for placement in group.placements
        if placement.polyline_id in kept_ids
    )
    return replace(group, polylines=tuple(kept), placements=placements)


def move_polylines_between_groups(
    source: RoadUsageGroupRecipe,
    dest: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
) -> tuple[RoadUsageGroupRecipe, RoadUsageGroupRecipe]:
    """move polylines by id from source to dest; placements for those ids are dropped"""
    move_ids: set[str] = set(polyline_ids)
    if not move_ids:
        return source, dest
    moving: list[RoadPolyline] = [
        polyline for polyline in source.polylines if polyline.id in move_ids
    ]
    if not moving:
        return source, dest
    moved_ids: set[str] = {polyline.id for polyline in moving}
    source_kept: list[RoadPolyline] = [
        polyline for polyline in source.polylines if polyline.id not in moved_ids
    ]
    source_placements = tuple(
        placement
        for placement in source.placements
        if placement.polyline_id not in moved_ids
    )
    dest_polylines: list[RoadPolyline] = [
        polyline for polyline in dest.polylines if polyline.id not in moved_ids
    ]
    dest_polylines.extend(moving)
    dest_placements = tuple(
        placement
        for placement in dest.placements
        if placement.polyline_id not in moved_ids
    )
    return (
        replace(
            source,
            polylines=tuple(source_kept),
            placements=source_placements,
        ),
        replace(
            dest,
            polylines=tuple(dest_polylines),
            placements=dest_placements,
        ),
    )


def normalize_usage_name(name: str) -> str:
    """strip and lowercase a usage label; empty raises"""
    cleaned: str = name.strip().lower()
    if not cleaned:
        raise ValueError("usage name is empty")
    return cleaned


def simplify_polylines(
    group: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
    eps_m: float,
) -> RoadUsageGroupRecipe:
    """douglas-peucker selected polylines; keep ids"""
    selected: set[str] = set(polyline_ids)
    updated: list[RoadPolyline] = []
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id not in selected:
            updated.append(polyline)
            continue
        simplified: list[tuple[float, float]] = simplify_polyline(
            list(polyline.points), eps_m
        )
        if len(simplified) < 2:
            updated.append(polyline)
            continue
        updated.append(replace(polyline, points=tuple(simplified)))
    return replace_usage_polylines(group, updated)


def endpoint_point(
    polyline: RoadPolyline, vertex_index: int
) -> tuple[float, float]:
    """xz at a vertex index"""
    return polyline.points[vertex_index]


def set_polyline_vertex_xz(
    group: RoadUsageGroupRecipe,
    polyline_id: str,
    vertex_index: int,
    x: float,
    z: float,
) -> RoadUsageGroupRecipe:
    """move one polyline vertex in xz; placements cleared by the drag commit"""
    found: RoadPolyline | None = polyline_by_id(group, polyline_id)
    if found is None:
        raise ValueError(f"unknown polyline {polyline_id!r}")
    if vertex_index < 0 or vertex_index >= len(found.points):
        raise ValueError(f"vertex index out of range: {vertex_index}")
    points: list[tuple[float, float]] = list(found.points)
    points[vertex_index] = (float(x), float(z))
    updated_line: RoadPolyline = replace(found, points=tuple(points))
    lines: list[RoadPolyline] = [
        updated_line if polyline.id == polyline_id else polyline
        for polyline in group.polylines
    ]
    return replace_usage_polylines(group, lines)


def endpoint_distance(
    a: RoadPolyline,
    a_index: int,
    b: RoadPolyline,
    b_index: int,
) -> float:
    """distance between two polyline vertices"""
    ax: float
    az: float
    ax, az = endpoint_point(a, a_index)
    bx: float
    bz: float
    bx, bz = endpoint_point(b, b_index)
    return hypot(ax - bx, az - bz)


def reverse_polyline(polyline: RoadPolyline) -> RoadPolyline:
    """reverse point order; keep id"""
    return replace(polyline, points=tuple(reversed(polyline.points)))


def try_join_pair(
    a: RoadPolyline,
    b: RoadPolyline,
    snap_m: float,
) -> RoadPolyline | None:
    """join two polylines when an end pair is within snap; new id"""
    ends_a: tuple[int, int] = (0, len(a.points) - 1)
    ends_b: tuple[int, int] = (0, len(b.points) - 1)
    best: tuple[float, int, int] | None = None
    ai: int
    bi: int
    for ai in ends_a:
        for bi in ends_b:
            dist: float = endpoint_distance(a, ai, b, bi)
            if dist > snap_m:
                continue
            if best is None or dist < best[0]:
                best = (dist, ai, bi)
    if best is None:
        return None
    _, a_end, b_end = best
    left: RoadPolyline = a
    right: RoadPolyline = b
    # orient so left ends at join and right starts at join
    if a_end == 0:
        left = reverse_polyline(a)
    if b_end != 0:
        right = reverse_polyline(b)
    joined_points: tuple[tuple[float, float], ...] = left.points + right.points[1:]
    if len(joined_points) < 2:
        return None
    return RoadPolyline(id=new_node_id(), points=joined_points)


def min_endpoint_distance(a: RoadPolyline, b: RoadPolyline) -> float:
    """smallest end-to-end distance between two polylines"""
    best: float = float("inf")
    ai: int
    bi: int
    for ai in (0, len(a.points) - 1):
        for bi in (0, len(b.points) - 1):
            dist: float = endpoint_distance(a, ai, b, bi)
            if dist < best:
                best = dist
    return best


def merge_polylines(
    group: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
    snap_m: float = road_endpoint_snap_m,
) -> tuple[RoadUsageGroupRecipe, list[str]]:
    """greedily join joinable pairs within the selection; leave the rest

    returns (updated group, ids of resulting selected polylines)
    """
    selected: list[RoadPolyline] = []
    polyline_id: str
    for polyline_id in polyline_ids:
        found: RoadPolyline | None = polyline_by_id(group, polyline_id)
        if found is None:
            raise ValueError(f"unknown polyline {polyline_id!r}")
        selected.append(found)
    if len(selected) < 2:
        raise ValueError("merge needs at least two polylines")
    pool: list[RoadPolyline] = list(selected)
    while True:
        best: tuple[float, int, int, RoadPolyline] | None = None
        i: int
        j: int
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                dist: float = min_endpoint_distance(pool[i], pool[j])
                if dist > snap_m:
                    continue
                joined: RoadPolyline | None = try_join_pair(
                    pool[i], pool[j], snap_m
                )
                if joined is None:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, i, j, joined)
        if best is None:
            break
        _, i, j, joined = best
        # remove higher index first
        pool.pop(j)
        pool.pop(i)
        pool.append(joined)
    keep_ids: set[str] = set(polyline_ids)
    others: list[RoadPolyline] = [
        polyline for polyline in group.polylines if polyline.id not in keep_ids
    ]
    result_ids: list[str] = [polyline.id for polyline in pool]
    surviving: set[str] = {polyline.id for polyline in others} | set(result_ids)
    placements = tuple(
        placement
        for placement in group.placements
        if placement.polyline_id in surviving
    )
    return (
        replace(
            group,
            polylines=tuple(others + pool),
            placements=placements,
        ),
        result_ids,
    )


def bridge_endpoints(
    group: RoadUsageGroupRecipe,
    a: RoadEndpointRef,
    b: RoadEndpointRef,
) -> RoadUsageGroupRecipe:
    """insert a new two-point polyline between two endpoints"""
    if a.polyline_id == b.polyline_id:
        raise ValueError("bridge needs endpoints on two different polylines")
    line_a: RoadPolyline | None = polyline_by_id(group, a.polyline_id)
    line_b: RoadPolyline | None = polyline_by_id(group, b.polyline_id)
    if line_a is None or line_b is None:
        raise ValueError("unknown polyline for bridge")
    if (
        a.vertex_index < 0
        or a.vertex_index >= len(line_a.points)
        or b.vertex_index < 0
        or b.vertex_index >= len(line_b.points)
    ):
        raise ValueError("bridge vertex index out of range")
    point_a: tuple[float, float] = endpoint_point(line_a, a.vertex_index)
    point_b: tuple[float, float] = endpoint_point(line_b, b.vertex_index)
    if hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]) < 1e-6:
        raise ValueError("bridge endpoints coincide")
    bridge: RoadPolyline = RoadPolyline(
        id=new_node_id(),
        points=(point_a, point_b),
    )
    return replace_usage_polylines(group, list(group.polylines) + [bridge])


def split_polyline_at_vertex(
    group: RoadUsageGroupRecipe,
    polyline_id: str,
    vertex_index: int,
) -> RoadUsageGroupRecipe:
    """split one polyline at an interior vertex into two sharing that point"""
    found: RoadPolyline | None = polyline_by_id(group, polyline_id)
    if found is None:
        raise ValueError(f"unknown polyline {polyline_id!r}")
    if vertex_index <= 0 or vertex_index >= len(found.points) - 1:
        raise ValueError("split needs an interior vertex")
    left: RoadPolyline = replace(
        found, points=tuple(found.points[: vertex_index + 1])
    )
    right: RoadPolyline = RoadPolyline(
        id=new_node_id(),
        points=tuple(found.points[vertex_index:]),
    )
    lines: list[RoadPolyline] = []
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id != polyline_id:
            lines.append(polyline)
            continue
        lines.append(left)
        lines.append(right)
    return drop_placements(
        replace_usage_polylines(group, lines), [polyline_id]
    )


def insert_vertex_between(
    group: RoadUsageGroupRecipe,
    a: RoadEndpointRef,
    b: RoadEndpointRef,
) -> tuple[RoadUsageGroupRecipe, RoadEndpointRef]:
    """insert the chord midpoint between two verts on the same polyline

    returns (updated group, ref to the new vertex)
    """
    if a.polyline_id != b.polyline_id:
        raise ValueError("insert needs two vertices on the same polyline")
    if a.vertex_index == b.vertex_index:
        raise ValueError("insert needs two distinct vertices")
    found: RoadPolyline | None = polyline_by_id(group, a.polyline_id)
    if found is None:
        raise ValueError(f"unknown polyline {a.polyline_id!r}")
    lo: int = min(a.vertex_index, b.vertex_index)
    hi: int = max(a.vertex_index, b.vertex_index)
    if lo < 0 or hi >= len(found.points):
        raise ValueError("insert vertex index out of range")
    ax: float
    az: float
    ax, az = found.points[lo]
    bx: float
    bz: float
    bx, bz = found.points[hi]
    mid: tuple[float, float] = ((ax + bx) * 0.5, (az + bz) * 0.5)
    insert_at: int = lo + 1
    points: list[tuple[float, float]] = list(found.points)
    points.insert(insert_at, mid)
    updated_line: RoadPolyline = replace(found, points=tuple(points))
    lines: list[RoadPolyline] = [
        updated_line if polyline.id == found.id else polyline
        for polyline in group.polylines
    ]
    updated: RoadUsageGroupRecipe = drop_placements(
        replace_usage_polylines(group, lines), [found.id]
    )
    new_ref: RoadEndpointRef = RoadEndpointRef(
        polyline_id=found.id, vertex_index=insert_at
    )
    return updated, new_ref


def set_usage_group_on_node(
    node: Node, group: RoadUsageGroupRecipe
) -> Node:
    """replace the road usage group recipe on a node"""
    if node.road_usage_group is None:
        raise ValueError("node is not a road usage group")
    return replace(node, road_usage_group=group)
