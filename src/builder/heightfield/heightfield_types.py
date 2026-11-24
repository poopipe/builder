"""heightfield recipe types"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from math import gcd

from builder.generators.param_types import EnumParam, FloatParam
from builder.heightfield.heightmap_catalog import HeightmapId

# ushort indices: tile_verts^2 must stay <= 65535
heightfield_tile_step: int = 128
heightfield_tile_verts: int = heightfield_tile_step + 1
# world size snaps to multiples of 256m (and to a spacing-compatible quantum)
heightfield_size_multiple: int = 256
# 32k m at 1 m spacing → 32768 / 128 = 256 tiles
heightfield_max_tiles_per_axis: int = 256


class VertexSpacing(IntEnum):
    """metres between adjacent heightfield vertices"""

    m1 = 1
    m2 = 2
    m4 = 4


class HeightBlendMode(IntEnum):
    """how an edit layer combines with the accumulated height"""

    add = 1
    subtract = 2
    multiply = 3
    max = 4
    min = 5


class HeightLayerSourceType(IntEnum):
    """where a layer height or mask bitmap comes from"""

    catalog = 1
    paint = 2
    procedural = 3
    sculpt = 4
    vector = 5
    vector_sculpt = 6
    vector_road = 7
    road_sections = 8


class SculptHeightMode(IntEnum):
    """how a sculpt correction layer stores and composites metres"""

    delta = 1
    absolute = 2


class ProceduralHeightType(IntEnum):
    """procedural bitmap generators (stub set for data model)"""

    constant = 1


@dataclass(frozen=True)
class CatalogHeightSource:
    """height or mask samples from a catalog heightmap"""

    heightmap_id: HeightmapId


@dataclass(frozen=True)
class PaintHeightSource:
    """mutable paint raster stored as a catalog heightmap id"""

    heightmap_id: HeightmapId


@dataclass(frozen=True)
class SculptHeightSource:
    """mutable sculpt correction raster (metres) over the stack below"""

    heightmap_id: HeightmapId
    mode: SculptHeightMode = SculptHeightMode.delta
    # absolute mode coverage 0..1; unused (none) in delta mode
    coverage_id: HeightmapId | None = None


@dataclass(frozen=True)
class ProceduralHeightSource:
    """procedural samples regenerated from params"""

    procedural_type: ProceduralHeightType = ProceduralHeightType.constant
    constant_value: float = 0.0


@dataclass(frozen=True)
class VectorHeightSource:
    """derived paint raster from a live VectorLayer node hierarchy"""

    layer_node_id: str
    heightmap_id: HeightmapId


@dataclass(frozen=True)
class VectorSculptHeightSource:
    """derived sculpt delta (metres) from a live VectorLayer node hierarchy"""

    layer_node_id: str
    heightmap_id: HeightmapId


@dataclass(frozen=True)
class VectorRoadHeightSource:
    """absolute ribbon height + coverage that lerps the stack below"""

    layer_node_id: str
    heightmap_id: HeightmapId
    coverage_id: HeightmapId


@dataclass(frozen=True)
class RoadSectionsHeightSource:
    """authorship layer for shp-imported road section splines (no heightmap)"""

    layer_node_id: str


HeightLayerSource = (
    CatalogHeightSource
    | PaintHeightSource
    | SculptHeightSource
    | ProceduralHeightSource
    | VectorHeightSource
    | VectorSculptHeightSource
    | VectorRoadHeightSource
    | RoadSectionsHeightSource
)


@dataclass(frozen=True)
class HeightEditLayer:
    """one non-destructive edit layer above the base heightmap"""

    name: str
    height: HeightLayerSource
    enabled: bool = True
    blend: HeightBlendMode = HeightBlendMode.add
    opacity: FloatParam = FloatParam(
        1.0, label="Opacity", step=0.05, minimum=0.0, maximum=1.0
    )
    mask: HeightLayerSource | None = None
    amplitude: float = 1.0
    offset: float = 0.0
    uv_scale_u: float = 1.0
    uv_scale_v: float = 1.0
    uv_offset_u: float = 0.0
    uv_offset_v: float = 0.0


@dataclass(frozen=True)
class Heightfield:
    """tiled heightfield recipe on a group node

    author size (metres) and vertex spacing; tiles are derived
    base is the immutable loaded heightmap; edit_layers blend on top
    """

    size_x: FloatParam = FloatParam(
        256.0,
        label="Size X",
        step=float(heightfield_size_multiple),
        minimum=float(heightfield_size_multiple),
    )
    size_z: FloatParam = FloatParam(
        256.0,
        label="Size Z",
        step=float(heightfield_size_multiple),
        minimum=float(heightfield_size_multiple),
    )
    spacing: EnumParam = EnumParam(VertexSpacing.m1, label="Spacing")
    # map-space lower-left (asc xllcorner / yllcorner); hf mesh stays centered
    origin_x: FloatParam = FloatParam(0.0, label="Origin X")
    origin_z: FloatParam = FloatParam(0.0, label="Origin Z")
    base_height_min: FloatParam = FloatParam(0.0, label="Height Min")
    base_height_max: FloatParam = FloatParam(1.0, label="Height Max")
    base: HeightmapId | None = None
    # absolute path at runtime; scene json stores project-relative
    albedo_path: str | None = None
    edit_layers: tuple[HeightEditLayer, ...] = ()


def map_world_xz_to_hf_local(
    world_x: float,
    world_z: float,
    *,
    origin_x: float,
    origin_z: float,
    size_x: float,
    size_z: float,
) -> tuple[float, float]:
    """map map-space xz (asc / shp) into centered heightfield-local xz

    east-west mirrored to match asc decode (buffer column 0 = east)
    """
    local_x: float = world_x - origin_x - 0.5 * size_x
    local_z: float = world_z - origin_z - 0.5 * size_z
    return (-local_x, local_z)


@dataclass(frozen=True)
class HeightfieldLayout:
    """derived tile grid for a normalized heightfield"""

    tiles_x: int
    tiles_z: int
    tile_verts: int
    res_x: int
    res_z: int
    spacing: float
    size_x: float
    size_z: float


def spacing_metres(heightfield: Heightfield) -> float:
    """vertex spacing in metres from the spacing enum"""
    return float(heightfield.spacing.value)


def size_snap_quantum(spacing: float) -> float:
    """smallest size step that is a multiple of 256 and tiles with step 128"""
    spacing_i: int = int(round(spacing))
    if spacing_i < 1:
        raise ValueError("spacing must be at least 1 metre")
    cell: int = heightfield_tile_step * spacing_i
    # lcm(256, cell) so size/spacing is divisible by tile_step
    return float(heightfield_size_multiple * cell // gcd(heightfield_size_multiple, cell))


def snap_heightfield_size(metres: float, spacing: float) -> float:
    """snap world size up/down to a valid quantum for this spacing"""
    quantum: float = size_snap_quantum(spacing)
    if metres < quantum:
        return quantum
    return round(metres / quantum) * quantum


def normalize_heightfield(heightfield: Heightfield) -> Heightfield:
    """snap sizes to valid quanta for the current spacing"""
    spacing: float = spacing_metres(heightfield)
    size_x: float = snap_heightfield_size(heightfield.size_x.value, spacing)
    size_z: float = snap_heightfield_size(heightfield.size_z.value, spacing)
    return replace(
        heightfield,
        size_x=replace(heightfield.size_x, value=size_x),
        size_z=replace(heightfield.size_z, value=size_z),
    )


def heightfield_layout(heightfield: Heightfield) -> HeightfieldLayout:
    """derive tile counts and resolution from size and spacing"""
    spacing: float = spacing_metres(heightfield)
    size_x: float = snap_heightfield_size(heightfield.size_x.value, spacing)
    size_z: float = snap_heightfield_size(heightfield.size_z.value, spacing)
    intervals_x: int = int(round(size_x / spacing))
    intervals_z: int = int(round(size_z / spacing))
    if intervals_x % heightfield_tile_step != 0 or intervals_z % heightfield_tile_step != 0:
        raise ValueError("heightfield size/spacing does not tile with step 128")
    tiles_x: int = intervals_x // heightfield_tile_step
    tiles_z: int = intervals_z // heightfield_tile_step
    if tiles_x < 1 or tiles_z < 1:
        raise ValueError("heightfield needs at least one tile per axis")
    if tiles_x > heightfield_max_tiles_per_axis or tiles_z > heightfield_max_tiles_per_axis:
        raise ValueError(
            f"heightfield exceeds {heightfield_max_tiles_per_axis} tiles per axis"
        )
    return HeightfieldLayout(
        tiles_x=tiles_x,
        tiles_z=tiles_z,
        tile_verts=heightfield_tile_verts,
        res_x=intervals_x + 1,
        res_z=intervals_z + 1,
        spacing=spacing,
        size_x=size_x,
        size_z=size_z,
    )


def heightfield_resolution(heightfield: Heightfield) -> tuple[int, int]:
    """return full-field vertex counts (x, z)"""
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    return (layout.res_x, layout.res_z)


def default_heightfield() -> Heightfield:
    """return a small flat heightfield for placement"""
    return normalize_heightfield(Heightfield())


def height_layer_source_type(source: HeightLayerSource) -> HeightLayerSourceType:
    """discriminant for a height layer source"""
    match source:
        case CatalogHeightSource():
            return HeightLayerSourceType.catalog
        case PaintHeightSource():
            return HeightLayerSourceType.paint
        case SculptHeightSource():
            return HeightLayerSourceType.sculpt
        case ProceduralHeightSource():
            return HeightLayerSourceType.procedural
        case VectorHeightSource():
            return HeightLayerSourceType.vector
        case VectorSculptHeightSource():
            return HeightLayerSourceType.vector_sculpt
        case VectorRoadHeightSource():
            return HeightLayerSourceType.vector_road
        case RoadSectionsHeightSource():
            return HeightLayerSourceType.road_sections


def catalog_ids_for_source(source: HeightLayerSource | None) -> tuple[HeightmapId, ...]:
    """catalog ids referenced by a height or mask source"""
    match source:
        case CatalogHeightSource(heightmap_id=heightmap_id):
            return (heightmap_id,)
        case PaintHeightSource(heightmap_id=heightmap_id):
            return (heightmap_id,)
        case SculptHeightSource(heightmap_id=heightmap_id, coverage_id=coverage_id):
            if coverage_id is None:
                return (heightmap_id,)
            return (heightmap_id, coverage_id)
        case VectorHeightSource(heightmap_id=heightmap_id):
            return (heightmap_id,)
        case VectorSculptHeightSource(heightmap_id=heightmap_id):
            return (heightmap_id,)
        case VectorRoadHeightSource(
            heightmap_id=heightmap_id, coverage_id=coverage_id
        ):
            return (heightmap_id, coverage_id)
        case ProceduralHeightSource() | RoadSectionsHeightSource() | None:
            return ()


def catalog_ids_for_heightfield(heightfield: Heightfield) -> tuple[HeightmapId, ...]:
    """all catalog heightmap ids referenced by base and edit layers"""
    ids: list[HeightmapId] = []
    if heightfield.base is not None:
        ids.append(heightfield.base)
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        ids.extend(catalog_ids_for_source(layer.height))
        ids.extend(catalog_ids_for_source(layer.mask))
    return tuple(ids)


def move_edit_layer(
    layers: tuple[HeightEditLayer, ...],
    index: int,
    delta: int,
) -> tuple[tuple[HeightEditLayer, ...], int] | None:
    """swap layer with neighbor; None if out of range (nothing before index 0 / base)"""
    if delta not in (-1, 1):
        return None
    if index < 0 or index >= len(layers):
        return None
    target: int = index + delta
    if target < 0 or target >= len(layers):
        return None
    items: list[HeightEditLayer] = list(layers)
    items[index], items[target] = items[target], items[index]
    return (tuple(items), target)
