"""road sections layer recipes and usage classification"""

from __future__ import annotations

from dataclasses import dataclass

# metres: merge/connect ends within this distance
road_endpoint_snap_m: float = 2.0
# usage label when layer/highway attributes are empty
unknown_road_usage: str = "unknown"


@dataclass(frozen=True)
class RoadSectionsLayerRecipe:
    """marker recipe on a road sections layer under a heightfield"""


@dataclass(frozen=True)
class RoadPolyline:
    """one open polyline in hf-local xz"""

    id: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RoadPiecePlacement:
    """one fitted road piece instance in hf-local xz"""

    polyline_id: str
    stem: str
    x: float
    z: float
    yaw_deg: float
    roll_deg: float = 0.0


@dataclass(frozen=True)
class RoadUsageGroupRecipe:
    """one usage class under a shapefile; holds compact polylines"""

    usage: str
    # dayz road kit id (asf1, city, …); defaults from usage_to_kit on import
    kit: str
    polylines: tuple[RoadPolyline, ...] = ()
    placements: tuple[RoadPiecePlacement, ...] = ()
    # when true, meshed pieces snap to the authored path / terrain shadow
    project_onto_shadow: bool = False
    # when true, stamp an absolute ribbon into the road-sections heightmaps
    ribbon_flatten: bool = False


# fallback ribbon stamp radius (= full kit width) when catalogue has no entry
road_sections_ribbon_radius_m: float = 6.0
# paint buffer edge cap (base can be 4k; ribbon does not need that density)
road_sections_raster_max: int = 2048


@dataclass(frozen=True)
class RoadShpRecipe:
    """one imported shapefile under a road sections layer"""

    source_path: str


@dataclass(frozen=True)
class RoadEndpointRef:
    """one vertex on a road polyline (usually an end)"""

    polyline_id: str
    vertex_index: int


def classify_road_usage(layer: str, highway: str) -> str:
    """map gtt layer / osm highway to a usage class label

    prefers a concrete gtt layer; ``unfiltered`` / empty falls through to highway.
    always returns a label so every polyline feature can import
    """
    layer_key: str = (layer or "").strip().lower()
    highway_key: str = (highway or "").strip().lower()
    if layer_key and layer_key != "unfiltered":
        return layer_key
    if highway_key:
        return highway_key
    if layer_key:
        return layer_key
    return unknown_road_usage


def road_usage_polyline_count(group: RoadUsageGroupRecipe) -> int:
    """polylines stored on one usage group"""
    return len(group.polylines)


def polyline_by_id(
    group: RoadUsageGroupRecipe, polyline_id: str
) -> RoadPolyline | None:
    """find one polyline by id"""
    polyline: RoadPolyline
    for polyline in group.polylines:
        if polyline.id == polyline_id:
            return polyline
    return None
