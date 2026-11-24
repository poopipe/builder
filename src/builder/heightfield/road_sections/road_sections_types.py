"""road sections layer recipes and usage classification"""

from __future__ import annotations

from dataclasses import dataclass

# gtt/osm usage class keys (not dayz kit ids)
KNOWN_USAGE_CLASSES: frozenset[str] = frozenset(
    {
        "provincial",
        "primary",
        "trunk",
        "secondary",
        "tertiary",
        "residential",
        "living_street",
        "unclassified",
        "service",
        "track",
        "trackunclassified",
        "softtrack",
        "dirtroad",
        "footway",
        "path",
        "pedestrian",
        "cycleway",
        "bridleway",
        "steps",
    }
)

SKIP_USAGE_CLASSES: frozenset[str] = frozenset(
    {
        "",
        "raceway",
        "unfiltered",
    }
)


@dataclass(frozen=True)
class RoadSectionsLayerRecipe:
    """marker recipe on a road sections layer under a heightfield"""


@dataclass(frozen=True)
class RoadPolyline:
    """one open polyline in hf-local xz"""

    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RoadUsageGroupRecipe:
    """one usage class under a shapefile; holds compact polylines"""

    usage: str
    polylines: tuple[RoadPolyline, ...] = ()


@dataclass(frozen=True)
class RoadShpRecipe:
    """one imported shapefile under a road sections layer"""

    source_path: str


def classify_road_usage(layer: str, highway: str) -> str | None:
    """map gtt layer / osm highway to a usage class, or none to skip

    prefers the gtt layer when it is a known road class; falls back to highway
    when the layer is empty, unfiltered, or otherwise unknown
    """
    layer_key: str = (layer or "").strip().lower()
    highway_key: str = (highway or "").strip().lower()
    if layer_key and layer_key not in SKIP_USAGE_CLASSES:
        if layer_key in KNOWN_USAGE_CLASSES:
            return layer_key
    if highway_key and highway_key not in SKIP_USAGE_CLASSES:
        if highway_key in KNOWN_USAGE_CLASSES:
            return highway_key
    return None


def road_usage_polyline_count(group: RoadUsageGroupRecipe) -> int:
    """polylines stored on one usage group"""
    return len(group.polylines)
