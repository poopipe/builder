"""place and clear road piece placements on usage-group polylines"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from pyray import Shader

from builder.generators.regenerate import mesh_child_ids
from builder.heightfield.heightfield_types import Heightfield, heightfield_layout
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.road_sections.fit import (
    FittedPiece,
    fit_polyline_pieces,
)
from builder.heightfield.road_sections.overlay import (
    find_road_sections_layer_ancestor,
    road_sections_layer_index,
    road_sections_plane_y,
)
from builder.heightfield.road_sections.part_catalog import (
    RoadPartsCatalog,
    resolve_kit,
)
from builder.heightfield.road_sections.proxies import (
    ensure_road_proxy,
    placement_transform,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
)
from builder.heightfield.vector.overlay import surface_y
from builder.heightfield.vector.rasterize import find_heightfield_group_for_descendant
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.view.prepare_mesh import PreparedMesh


def road_piece_child_id(
    usage_group_id: str, polyline_id: str, index: int
) -> str:
    """deterministic mesh child id for a placement slot"""
    return f"{usage_group_id}__piece__{polyline_id}__{index}"


def placements_for_polyline(
    group: RoadUsageGroupRecipe, polyline_id: str
) -> tuple[RoadPiecePlacement, ...]:
    """placements belonging to one polyline"""
    return tuple(
        placement
        for placement in group.placements
        if placement.polyline_id == polyline_id
    )


def drop_placements(
    group: RoadUsageGroupRecipe, polyline_ids: Sequence[str]
) -> RoadUsageGroupRecipe:
    """remove placements for the given polyline ids"""
    drop: set[str] = set(polyline_ids)
    kept: tuple[RoadPiecePlacement, ...] = tuple(
        placement
        for placement in group.placements
        if placement.polyline_id not in drop
    )
    return replace(group, placements=kept)


def place_polylines(
    group: RoadUsageGroupRecipe,
    polyline_ids: Sequence[str],
    catalog: RoadPartsCatalog,
) -> RoadUsageGroupRecipe:
    """fit kit pieces onto selected polylines; replace their placements"""
    catalog.ensure_loaded()
    kit = resolve_kit(group.kit, catalog.kits)

    selected: set[str] = set(polyline_ids)
    kept: list[RoadPiecePlacement] = [
        placement
        for placement in group.placements
        if placement.polyline_id not in selected
    ]
    polyline_id: str
    for polyline_id in polyline_ids:
        polyline: RoadPolyline | None = polyline_by_id(group, polyline_id)
        if polyline is None:
            raise ValueError(f"unknown polyline {polyline_id!r}")
        fitted: list[FittedPiece] = fit_polyline_pieces(
            list(polyline.points), kit
        )
        piece: FittedPiece
        for piece in fitted:
            kept.append(
                RoadPiecePlacement(
                    polyline_id=polyline_id,
                    stem=piece.stem,
                    x=piece.x,
                    z=piece.z,
                    yaw_deg=piece.yaw_deg,
                    roll_deg=piece.roll_deg,
                )
            )
    return replace(group, placements=tuple(kept))


def project_placement_to_shadow(
    placement: RoadPiecePlacement,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    size_x: float,
    size_z: float,
    plane_y: float,
    *,
    below_layer_index: int | None = None,
) -> tuple[float, float, float]:
    """sample underlay y at the fit placement xz; keep fit x/z/yaw

    y_local is relative to the road-sections work plane so composed hf y
    matches the shadow (below this road-sections layer).
    """
    surface: float = surface_y(
        catalog,
        heightfield,
        placement.x,
        placement.z,
        size_x,
        size_z,
        0.0,
        below_layer_index=below_layer_index,
    )
    y_local: float = surface - plane_y
    return placement.x, y_local, placement.z


def regenerate_road_piece_children(
    scene: Scene,
    usage_group_id: str,
    catalog: RoadPartsCatalog,
    *,
    has_mesh: Callable[[MeshId], bool],
    register_mesh: Callable[[MeshId, PreparedMesh], None],
    shader: Shader,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> None:
    """rebuild meshed piece children from the usage group's placements"""
    group_node: Node | None = scene.nodes.get(usage_group_id)
    if group_node is None or group_node.road_usage_group is None:
        return
    group: RoadUsageGroupRecipe = group_node.road_usage_group
    scene.remove_nodes(mesh_child_ids(scene.nodes, usage_group_id))

    heightfield: Heightfield | None = None
    size_x: float = 0.0
    size_z: float = 0.0
    plane_y: float = 0.0
    below_layer_index: int | None = None
    if group.project_onto_shadow and heightmap_catalog is not None:
        hf_id: str | None = find_heightfield_group_for_descendant(
            scene.nodes, usage_group_id
        )
        if hf_id is not None:
            hf_node: Node | None = scene.nodes.get(hf_id)
            if hf_node is not None and hf_node.heightfield is not None:
                heightfield = hf_node.heightfield
                layout = heightfield_layout(heightfield)
                size_x = layout.size_x
                size_z = layout.size_z
                layer_id: str | None = find_road_sections_layer_ancestor(
                    scene.nodes, usage_group_id
                )
                if layer_id is not None:
                    plane_y = road_sections_plane_y(
                        scene.nodes, layer_id, hf_id
                    )
                    below_layer_index = road_sections_layer_index(
                        heightfield, layer_id
                    )

    # stable order: by polyline id then existing placement order
    by_polyline: dict[str, list[RoadPiecePlacement]] = {}
    placement: RoadPiecePlacement
    for placement in group.placements:
        by_polyline.setdefault(placement.polyline_id, []).append(placement)

    children: list[Node] = []
    polyline_id: str
    for polyline_id in sorted(by_polyline.keys()):
        row: list[RoadPiecePlacement] = by_polyline[polyline_id]
        index: int
        for index, placement in enumerate(row):
            mesh_id: MeshId = ensure_road_proxy(
                catalog,
                placement.stem,
                width_m=catalog.width_for_kit(group.kit),
                has_mesh=has_mesh,
                register_mesh=register_mesh,
                shader=shader,
            )
            x: float = placement.x
            y: float = 0.0
            z: float = placement.z
            if (
                group.project_onto_shadow
                and heightfield is not None
                and heightmap_catalog is not None
            ):
                x, y, z = project_placement_to_shadow(
                    placement,
                    heightmap_catalog,
                    heightfield,
                    size_x,
                    size_z,
                    plane_y,
                    below_layer_index=below_layer_index,
                )
            children.append(
                Node(
                    id=road_piece_child_id(
                        usage_group_id, polyline_id, index
                    ),
                    parent_id=usage_group_id,
                    transform=placement_transform(
                        x,
                        z,
                        placement.yaw_deg,
                        placement.roll_deg,
                        y=y,
                        pitch_deg=0.0,
                    ),
                    mesh_id=mesh_id,
                )
            )
    if children:
        scene.add_nodes(children)
