"""low-level paint / sculpt stamps used by vector shape rasterizers"""

from __future__ import annotations

from math import hypot

from builder.heightfield.brushes.brush_types import (
    BrushStrokeContext,
    brush_falloff,
    brush_falloff_flatten,
    clamp_paint_sample,
)
from builder.heightfield.brushes.registry import get_brush_spec
from builder.heightfield.vector.vector_types import (
    VectorRasterContext,
    radius_metres_to_uv,
    world_xz_to_uv,
)


def is_flatten_brush(brush_kind: str) -> bool:
    """true for flatten / flatten_up / flatten_down"""
    return brush_kind in ("flatten", "flatten_up", "flatten_down")


def apply_sculpt_weight(
    context: VectorRasterContext,
    index: int,
    u: float,
    v: float,
    weight: float,
) -> None:
    """apply one weighted sculpt-metre contribution at a buffer index"""
    if weight <= 0.0:
        return
    current: float = context.samples[index]
    kind: str = context.brush_kind
    if kind == "add":
        context.samples[index] = current + context.offset * weight
        return
    if is_flatten_brush(kind):
        if context.flatten_target is None or context.below_at_uv is None:
            return
        below: float = context.below_at_uv(u, v)
        visible: float = below + current
        target: float = float(context.flatten_target)
        if kind == "flatten_up" and visible >= target:
            return
        if kind == "flatten_down" and visible <= target:
            return
        desired: float = target - below
        mix: float = weight * max(0.0, min(1.0, float(context.intensity)))
        context.samples[index] = current + (desired - current) * mix
        return
    # smooth and unknown kinds: no-op on sculpt vector


def world_aabb_to_texel_range(
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    size_x: float,
    size_z: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """inclusive-exclusive texel ranges covering a world xz aabb; None if empty"""
    if width < 1 or height < 1 or size_x <= 1e-8 or size_z <= 1e-8:
        return None
    u0: float = (min_x + 0.5 * size_x) / size_x
    u1: float = (max_x + 0.5 * size_x) / size_x
    v0: float = (min_z + 0.5 * size_z) / size_z
    v1: float = (max_z + 0.5 * size_z) / size_z
    x0: int = max(0, int(u0 * float(width) - 0.5))
    x1: int = min(width, int(u1 * float(width) + 0.5) + 1)
    y0: int = max(0, int(v0 * float(height) - 0.5))
    y1: int = min(height, int(v1 * float(height) + 0.5) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, x1, y0, y1)


def texel_range_to_world_aabb(
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    size_x: float,
    size_z: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """world xz aabb covering an inclusive-exclusive texel range"""
    min_x: float = (float(x0) / float(width)) * size_x - 0.5 * size_x
    max_x: float = (float(x1) / float(width)) * size_x - 0.5 * size_x
    min_z: float = (float(y0) / float(height)) * size_z - 0.5 * size_z
    max_z: float = (float(y1) / float(height)) * size_z - 0.5 * size_z
    return (min_x, max_x, min_z, max_z)


def clip_texel_range(
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    context: VectorRasterContext,
) -> tuple[int, int, int, int] | None:
    """intersect a texel range with context.clip_*; None if empty"""
    cx0: int = x0
    cx1: int = x1
    cy0: int = y0
    cy1: int = y1
    if context.clip_x0 is not None:
        assert context.clip_x1 is not None
        assert context.clip_y0 is not None
        assert context.clip_y1 is not None
        cx0 = max(cx0, context.clip_x0)
        cx1 = min(cx1, context.clip_x1)
        cy0 = max(cy0, context.clip_y0)
        cy1 = min(cy1, context.clip_y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    return (cx0, cx1, cy0, cy1)


def apply_road_absolute(
    context: VectorRasterContext,
    index: int,
    target_height: float,
    weight: float,
) -> None:
    """write premultiplied absolute height (h*cov) and coverage for one texel

    premultiplied storage keeps bilinear upsample of abs+cov filters correct
    when exterior samples are cleared to 0
    """
    if weight <= 0.0 or context.coverage_samples is None:
        return
    mix: float = weight * max(0.0, min(1.0, float(context.intensity)))
    if mix <= 0.0:
        return
    cov: float = max(0.0, min(1.0, context.coverage_samples[index]))
    height: float = float(target_height)
    if cov <= 1e-8:
        cov_new: float = mix
        context.samples[index] = height * cov_new
        context.coverage_samples[index] = cov_new
        return
    stored_height: float = context.samples[index] / cov
    height_new: float = stored_height + (height - stored_height) * mix
    cov_new = cov + (1.0 - cov) * mix
    context.samples[index] = height_new * cov_new
    context.coverage_samples[index] = cov_new


def apply_road_weight(
    context: VectorRasterContext,
    index: int,
    u: float,
    v: float,
    weight: float,
) -> None:
    """write absolute ribbon height and coverage for a road texel

    ribbon_height_at_xz receives below=0; absolute grade is precomputed by the
    caller (no underlay sample here)
    """
    if weight <= 0.0:
        return
    if (
        context.ribbon_height_at_xz is None
        or context.coverage_samples is None
    ):
        return
    wx: float = (u - 0.5) * context.size_x
    wz: float = (v - 0.5) * context.size_z
    target: float = float(context.ribbon_height_at_xz(wx, wz, 0.0))
    apply_road_absolute(context, index, target, weight)


def apply_weighted_sample(
    context: VectorRasterContext,
    index: int,
    u: float,
    v: float,
    weight: float,
) -> None:
    """paint or sculpt contribution for fill / soft-band paths"""
    if weight <= 0.0:
        return
    if context.ribbon_height_at_xz is not None:
        apply_road_weight(context, index, u, v, weight)
        return
    if context.sculpt_metres:
        apply_sculpt_weight(context, index, u, v, weight)
        return
    context.samples[index] = clamp_paint_sample(
        context.samples[index] + context.offset * weight
    )


def stamp_sculpt_brush_at_uv(
    context: VectorRasterContext, u: float, v: float
) -> None:
    """metre stamp disk at field uv (sculpt vector path)"""
    radius_u: float
    radius_v: float
    radius_u, radius_v = radius_metres_to_uv(
        context.radius_metres, context.size_x, context.size_z
    )
    stroke: BrushStrokeContext = BrushStrokeContext(
        center_u=u,
        center_v=v,
        radius_u=radius_u,
        radius_v=radius_v,
        intensity=context.offset,
        flatten_target=context.flatten_target,
    )
    width: int = context.width
    height: int = context.height
    if width < 1 or height < 1:
        return
    radius_px_u: float = radius_u * float(width)
    radius_px_v: float = radius_v * float(height)
    cx: float = u * float(width)
    cy: float = v * float(height)
    x0: int = max(0, int(cx - radius_px_u) - 1)
    x1: int = min(width - 1, int(cx + radius_px_u) + 1)
    y0: int = max(0, int(cy - radius_px_v) - 1)
    y1: int = min(height - 1, int(cy + radius_px_v) + 1)
    y: int
    for y in range(y0, y1 + 1):
        x: int
        for x in range(x0, x1 + 1):
            su: float = (float(x) + 0.5) / float(width)
            sv: float = (float(y) + 0.5) / float(height)
            w: float = (
                brush_falloff_flatten(su, sv, stroke)
                if is_flatten_brush(context.brush_kind)
                else brush_falloff(su, sv, stroke)
            )
            if w <= 0.0:
                continue
            apply_sculpt_weight(
                context, y * width + x, su, sv, w
            )


def stamp_brush_at_uv(context: VectorRasterContext, u: float, v: float) -> None:
    """apply the bound brush stamp at one field uv"""
    if context.sculpt_metres:
        stamp_sculpt_brush_at_uv(context, u, v)
        return
    radius_u: float
    radius_v: float
    radius_u, radius_v = radius_metres_to_uv(
        context.radius_metres, context.size_x, context.size_z
    )
    stroke: BrushStrokeContext = BrushStrokeContext(
        center_u=u,
        center_v=v,
        radius_u=radius_u,
        radius_v=radius_v,
        intensity=context.offset,
        flatten_target=0.0,
    )
    spec = get_brush_spec(context.brush_kind)
    params = spec.params_type()
    spec.stamp(
        context.samples, context.width, context.height, stroke, params
    )


def stamp_brush_at_xz(context: VectorRasterContext, x: float, z: float) -> None:
    """apply the bound brush stamp at heightfield-local xz"""
    u: float
    v: float
    u, v = world_xz_to_uv(x, z, context.size_x, context.size_z)
    stamp_brush_at_uv(context, u, v)


def soft_band_weight(signed_dist_metres: float, radius_metres: float) -> float:
    """1 deep inside, 0 outside the soft band; smooth across ±radius"""
    if radius_metres <= 1e-8:
        return 1.0 if signed_dist_metres <= 0.0 else 0.0
    # signed_dist: negative inside, positive outside
    if signed_dist_metres <= -radius_metres:
        return 1.0
    if signed_dist_metres >= radius_metres:
        return 0.0
    t: float = 0.5 - 0.5 * (signed_dist_metres / radius_metres)
    return t * t * (3.0 - 2.0 * t)


def stroke_weight(dist_metres: float, radius_metres: float) -> float:
    """1 on the path, 0 at brush radius; smoothstep matching brush disks"""
    if radius_metres <= 1e-8:
        return 1.0 if dist_metres <= 1e-8 else 0.0
    if dist_metres >= radius_metres:
        return 0.0
    t: float = 1.0 - dist_metres / radius_metres
    return t * t * (3.0 - 2.0 * t)


def ribbon_plateau_weight(
    dist_metres: float, half_width_metres: float
) -> float:
    """1 across the ribbon interior, soft only at the outer lip"""
    if half_width_metres <= 1e-8:
        return 1.0 if dist_metres <= 1e-8 else 0.0
    edge_soft: float = max(half_width_metres * 0.25, 0.5)
    # signed distance to the half-width boundary (negative inside)
    return soft_band_weight(dist_metres - half_width_metres, edge_soft)


def path_stroke_weight(
    context: VectorRasterContext,
    dist_metres: float,
    *,
    x: float = 0.0,
    z: float = 0.0,
) -> float:
    """plateau weight for roads; peaked stroke weight otherwise"""
    if context.ribbon_half_width_at_xz is not None:
        return ribbon_plateau_weight(
            dist_metres, context.ribbon_half_width_at_xz(x, z)
        )
    if context.ribbon_height_at_xz is not None:
        return ribbon_plateau_weight(dist_metres, context.radius_metres)
    return stroke_weight(dist_metres, context.radius_metres)


def point_in_polygon(x: float, z: float, points: list[tuple[float, float]]) -> bool:
    """even-odd point-in-polygon test in xz"""
    inside: bool = False
    n: int = len(points)
    if n < 3:
        return False
    j: int = n - 1
    i: int
    for i in range(n):
        xi: float
        zi: float
        xj: float
        zj: float
        xi, zi = points[i]
        xj, zj = points[j]
        intersects: bool = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / (zj - zi + 1e-30) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_to_segment_list(
    x: float,
    z: float,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    """minimum distance from xz to an explicit segment list"""
    if len(segments) == 0:
        return 1e9
    best: float = 1e9
    segment: tuple[tuple[float, float], tuple[float, float]]
    for segment in segments:
        ax: float
        az: float
        bx: float
        bz: float
        (ax, az), (bx, bz) = segment
        dx: float = bx - ax
        dz: float = bz - az
        length_sq: float = dx * dx + dz * dz
        t: float = 0.0
        if length_sq > 1e-12:
            t = max(
                0.0,
                min(1.0, ((x - ax) * dx + (z - az) * dz) / length_sq),
            )
        px: float = ax + dx * t
        pz: float = az + dz * t
        best = min(best, hypot(x - px, z - pz))
    return best


def distance_to_segments(
    x: float, z: float, points: list[tuple[float, float]], closed: bool
) -> float:
    """minimum distance from xz to polyline segments"""
    if len(points) == 0:
        return 1e9
    if len(points) == 1:
        return hypot(x - points[0][0], z - points[0][1])
    return distance_to_segment_list(
        x, z, polyline_segments(points, closed)
    )


def polyline_segments(
    points: list[tuple[float, float]], closed: bool
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """ordered open or closed segments from densified points"""
    if len(points) < 2:
        return []
    segment_count: int = len(points) if closed else len(points) - 1
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    i: int
    for i in range(segment_count):
        segments.append((points[i], points[(i + 1) % len(points)]))
    return segments


def segments_intersecting_aabb(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    pad: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """keep segments whose padded aabb overlaps the query rect"""
    kept: list[tuple[tuple[float, float], tuple[float, float]]] = []
    query_min_x: float = min_x - pad
    query_max_x: float = max_x + pad
    query_min_z: float = min_z - pad
    query_max_z: float = max_z + pad
    segment: tuple[tuple[float, float], tuple[float, float]]
    for segment in segments:
        (ax, az), (bx, bz) = segment
        seg_min_x: float = min(ax, bx)
        seg_max_x: float = max(ax, bx)
        seg_min_z: float = min(az, bz)
        seg_max_z: float = max(az, bz)
        if (
            seg_max_x < query_min_x
            or seg_min_x > query_max_x
            or seg_max_z < query_min_z
            or seg_min_z > query_max_z
        ):
            continue
        kept.append(segment)
    return kept


def stamp_world_aabb_for_clip(
    context: VectorRasterContext,
    fallback_min_x: float,
    fallback_max_x: float,
    fallback_min_z: float,
    fallback_max_z: float,
) -> tuple[float, float, float, float]:
    """world xz aabb covering the active clip rect, or the fallback stamp aabb"""
    if (
        context.clip_x0 is None
        or context.clip_x1 is None
        or context.clip_y0 is None
        or context.clip_y1 is None
    ):
        return (
            fallback_min_x,
            fallback_max_x,
            fallback_min_z,
            fallback_max_z,
        )
    return texel_range_to_world_aabb(
        context.clip_x0,
        context.clip_x1,
        context.clip_y0,
        context.clip_y1,
        context.size_x,
        context.size_z,
        context.width,
        context.height,
    )
