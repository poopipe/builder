"""terrain builder object-import .txt from road-sections placements"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from builder.heightfield.heightfield_types import (
    Heightfield,
    hf_local_xz_to_map_world,
)
from builder.heightfield.road_sections.part_catalog import (
    PieceType,
    parse_stem,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadUsageGroupRecipe,
)
from builder.heightfield.vector.coords import ordered_child_nodes
from builder.scene.scene_types import Node
from builder.tb_object import TbObject, write_object_import


@dataclass(frozen=True)
class TbRoadsExportResult:
    """summary of a terrain builder roads export"""

    objects_written: int
    files: tuple[Path, ...]


def corner_angle_for_stem(stem: str) -> float | None:
    """corner/curve angle degrees from stem, or none for straights"""
    parsed = parse_stem(stem)
    if parsed is None:
        return None
    if parsed.piece_type not in (PieceType.corner, PieceType.curve):
        return None
    return parsed.angle_deg


def mapframe_yaw_deg(
    local_yaw_deg: float,
    *,
    corner_angle_deg: float | None = None,
) -> float:
    """hf-local yaw → mapframe yaw

    east-west mirror flips heading sign. for corners it also flips bend sense,
    so add the editor's y-spin offset (180 - angle); roll stays 0.
    """
    yaw: float = (-float(local_yaw_deg)) % 360.0
    if corner_angle_deg is not None:
        yaw = (yaw + 180.0 - float(corner_angle_deg)) % 360.0
    return yaw


def placement_to_tb_object(
    placement: RoadPiecePlacement,
    *,
    origin_x: float,
    origin_z: float,
    size_x: float,
    size_z: float,
) -> TbObject:
    """one placement in terrain builder mapframe metres

    corners keep roll 0; chirality under the mirror is a yaw y-spin only.
    """
    world_x: float
    world_z: float
    world_x, world_z = hf_local_xz_to_map_world(
        placement.x,
        placement.z,
        origin_x=origin_x,
        origin_z=origin_z,
        size_x=size_x,
        size_z=size_z,
    )
    return TbObject(
        model=placement.stem,
        x=world_x,
        y=world_z,
        yaw=mapframe_yaw_deg(
            placement.yaw_deg,
            corner_angle_deg=corner_angle_for_stem(placement.stem),
        ),
        pitch=0.0,
        # chirality is encoded in yaw y-spin; never export roll 180
        roll=0.0,
    )


def collect_tb_objects_by_kit(
    nodes: dict[str, Node],
    layer_id: str,
    heightfield: Heightfield,
) -> dict[str, list[TbObject]]:
    """group placements under a road-sections layer by kit name"""
    origin_x: float = float(heightfield.origin_x.value)
    origin_z: float = float(heightfield.origin_z.value)
    size_x: float = float(heightfield.size_x.value)
    size_z: float = float(heightfield.size_z.value)
    by_kit: dict[str, list[TbObject]] = {}
    shp: Node
    for shp in ordered_child_nodes(nodes, layer_id):
        if shp.road_shp is None:
            continue
        usage: Node
        for usage in ordered_child_nodes(nodes, shp.id):
            group: RoadUsageGroupRecipe | None = usage.road_usage_group
            if group is None or len(group.placements) == 0:
                continue
            bucket: list[TbObject] = by_kit.setdefault(group.kit, [])
            placement: RoadPiecePlacement
            for placement in group.placements:
                bucket.append(
                    placement_to_tb_object(
                        placement,
                        origin_x=origin_x,
                        origin_z=origin_z,
                        size_x=size_x,
                        size_z=size_z,
                    )
                )
    return by_kit


def export_tb_road_imports(
    nodes: dict[str, Node],
    layer_id: str,
    heightfield: Heightfield,
    out_dir: Path,
) -> TbRoadsExportResult:
    """write import_<kit>.txt files for terrain builder object import"""
    out: Path = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_kit: dict[str, list[TbObject]] = collect_tb_objects_by_kit(
        nodes, layer_id, heightfield
    )
    files: list[Path] = []
    objects_written: int = 0
    kit_name: str
    objects: list[TbObject]
    for kit_name, objects in sorted(by_kit.items()):
        if not objects:
            continue
        path: Path = out / f"import_{kit_name}.txt"
        write_object_import(path, objects)
        files.append(path)
        objects_written += len(objects)
    return TbRoadsExportResult(
        objects_written=objects_written,
        files=tuple(files),
    )
