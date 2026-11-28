"""rectangle vector shape — two opposite corner points"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Vector3, vector3_transform

from builder.heightfield.vector.coords import (
    node_hf_local_matrix,
    ordered_child_nodes,
)
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    clip_texel_range,
    distance_to_segments,
    path_stroke_weight,
    point_in_polygon,
    soft_band_weight,
    world_aabb_to_texel_range,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorRasterContext,
    VectorShapeSpec,
)
from builder.scene.scene_types import Node, NodeRole


@dataclass(frozen=True)
class RectangleShapeParams:
    """kind-specific params for rectangle (geometry from two corners)"""


def rectangle_corners(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """axis-aligned corners from two opposite points in the same xz space"""
    if len(points) < 2:
        return None
    ax: float
    az: float
    bx: float
    bz: float
    ax, az = points[0]
    bx, bz = points[1]
    min_x: float = min(ax, bx)
    max_x: float = max(ax, bx)
    min_z: float = min(az, bz)
    max_z: float = max(az, bz)
    return [
        (min_x, min_z),
        (max_x, min_z),
        (max_x, max_z),
        (min_x, max_z),
    ]


def closed_rectangle_corners(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """four perimeter corners: pass-through if already four, else aabb of two"""
    if len(points) >= 4:
        return [
            (float(points[0][0]), float(points[0][1])),
            (float(points[1][0]), float(points[1][1])),
            (float(points[2][0]), float(points[2][1])),
            (float(points[3][0]), float(points[3][1])),
        ]
    return rectangle_corners(points)


def shape_local_path_point_xz(
    nodes: dict[str, Node], shape_id: str
) -> list[tuple[float, float]]:
    """path_point children as parent-local xz (shape-local for direct children)"""
    points: list[tuple[float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.path_point:
            continue
        points.append(
            (
                float(child.transform.translation.x),
                float(child.transform.translation.z),
            )
        )
    return points


def oriented_rectangle_hf_corners(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[float, float]] | None:
    """four hf-local corners from shape-local aabb of opposite path points"""
    local_points: list[tuple[float, float]] = shape_local_path_point_xz(
        nodes, shape_id
    )
    local_corners: list[tuple[float, float]] | None = rectangle_corners(
        local_points
    )
    if local_corners is None:
        return None
    shape_hf = node_hf_local_matrix(nodes, shape_id, heightfield_group_id)
    corners: list[tuple[float, float]] = []
    x: float
    z: float
    for x, z in local_corners:
        world_local: Vector3 = vector3_transform(Vector3(x, 0.0, z), shape_hf)
        corners.append((float(world_local.x), float(world_local.z)))
    return corners


def oriented_rectangle_hf_xzy(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[float, float, float]] | None:
    """oriented hf corners with parent-local y lerped from the two opposite points"""
    local_xzy: list[tuple[float, float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.path_point:
            continue
        t = child.transform.translation
        local_xzy.append((float(t.x), float(t.z), float(t.y)))
    if len(local_xzy) < 2:
        return None
    ax: float
    az: float
    ay: float
    bx: float
    bz: float
    by: float
    ax, az, ay = local_xzy[0]
    bx, bz, by = local_xzy[1]
    local_corners: list[tuple[float, float]] | None = rectangle_corners(
        [(ax, az), (bx, bz)]
    )
    if local_corners is None:
        return None
    span_x: float = bx - ax
    span_z: float = bz - az
    shape_hf = node_hf_local_matrix(nodes, shape_id, heightfield_group_id)
    result: list[tuple[float, float, float]] = []
    x: float
    z: float
    for x, z in local_corners:
        tx: float = 0.5
        tz: float = 0.5
        if abs(span_x) > 1e-8:
            tx = max(0.0, min(1.0, (x - ax) / span_x))
        if abs(span_z) > 1e-8:
            tz = max(0.0, min(1.0, (z - az) / span_z))
        y: float = ay + (by - ay) * (0.5 * (tx + tz))
        world_local: Vector3 = vector3_transform(Vector3(x, 0.0, z), shape_hf)
        result.append((float(world_local.x), float(world_local.z), y))
    return result


def rectangle_rasterize(
    context: VectorRasterContext,
    points: list[tuple[float, float]],
    _: RectangleShapeParams,
) -> None:
    """stroke perimeter or fill from two opposite corners or four oriented corners"""
    corners: list[tuple[float, float]] | None = closed_rectangle_corners(points)
    if corners is None:
        return
    min_x: float = min(p[0] for p in corners)
    max_x: float = max(p[0] for p in corners)
    min_z: float = min(p[1] for p in corners)
    max_z: float = max(p[1] for p in corners)
    pad: float = context.radius_metres
    ranges: tuple[int, int, int, int] | None = world_aabb_to_texel_range(
        min_x - pad,
        max_x + pad,
        min_z - pad,
        max_z + pad,
        context.size_x,
        context.size_z,
        context.width,
        context.height,
    )
    if ranges is None:
        return
    clipped: tuple[int, int, int, int] | None = clip_texel_range(
        ranges[0], ranges[1], ranges[2], ranges[3], context
    )
    if clipped is None:
        return
    x0: int
    x1: int
    y0: int
    y1: int
    x0, x1, y0, y1 = clipped
    inv_w: float = 1.0 / float(context.width)
    inv_h: float = 1.0 / float(context.height)
    match context.apply_mode:
        case VectorApplyMode.stroke:
            y: int
            for y in range(y0, y1):
                v: float = (float(y) + 0.5) * inv_h
                wz: float = (v - 0.5) * context.size_z
                x: int
                for x in range(x0, x1):
                    u: float = (float(x) + 0.5) * inv_w
                    wx: float = (u - 0.5) * context.size_x
                    dist: float = distance_to_segments(
                        wx, wz, corners, closed=True
                    )
                    weight: float = path_stroke_weight(
                        context, dist, x=wx, z=wz
                    )
                    if weight <= 0.0:
                        continue
                    apply_weighted_sample(
                        context,
                        y * context.width + x,
                        u,
                        v,
                        weight,
                    )
        case VectorApplyMode.fill:
            for y in range(y0, y1):
                v = (float(y) + 0.5) * inv_h
                wz = (v - 0.5) * context.size_z
                for x in range(x0, x1):
                    u = (float(x) + 0.5) * inv_w
                    wx = (u - 0.5) * context.size_x
                    dist = distance_to_segments(wx, wz, corners, closed=True)
                    inside: bool = point_in_polygon(wx, wz, corners)
                    signed: float = -dist if inside else dist
                    weight = soft_band_weight(signed, context.radius_metres)
                    if weight <= 0.0:
                        continue
                    apply_weighted_sample(
                        context,
                        y * context.width + x,
                        u,
                        v,
                        weight,
                    )


rectangle_shape_spec: VectorShapeSpec = VectorShapeSpec(
    kind="rectangle",
    label="Rectangle",
    params_type=RectangleShapeParams,
    default_apply_mode=VectorApplyMode.fill,
    rasterize=rectangle_rasterize,
)
