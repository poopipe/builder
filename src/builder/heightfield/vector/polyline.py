"""polyline vector shape — stroke along path or fill when closed"""

from __future__ import annotations

from dataclasses import dataclass

from builder.generators.param_types import BoolParam
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
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


@dataclass(frozen=True)
class PolylineShapeParams:
    """kind-specific params for polyline"""

    closed: BoolParam = BoolParam(False, label="Closed")


def polyline_stroke_sdf(
    context: VectorRasterContext,
    points: list[tuple[float, float]],
    closed: bool,
) -> None:
    """ribbon SDF along the polyline (stroke / open-fill)"""
    pad: float = context.radius_metres
    min_x: float = min(p[0] for p in points) - pad
    max_x: float = max(p[0] for p in points) + pad
    min_z: float = min(p[1] for p in points) - pad
    max_z: float = max(p[1] for p in points) + pad
    ranges: tuple[int, int, int, int] | None = world_aabb_to_texel_range(
        min_x,
        max_x,
        min_z,
        max_z,
        context.size_x,
        context.size_z,
        context.width,
        context.height,
    )
    if ranges is None:
        return
    x0: int
    x1: int
    y0: int
    y1: int
    x0, x1, y0, y1 = ranges
    inv_w: float = 1.0 / float(context.width)
    inv_h: float = 1.0 / float(context.height)
    y: int
    for y in range(y0, y1):
        v: float = (float(y) + 0.5) * inv_h
        wz: float = (v - 0.5) * context.size_z
        x: int
        for x in range(x0, x1):
            u: float = (float(x) + 0.5) * inv_w
            wx: float = (u - 0.5) * context.size_x
            dist: float = distance_to_segments(wx, wz, points, closed)
            weight: float = path_stroke_weight(context, dist, x=wx, z=wz)
            if weight <= 0.0:
                continue
            apply_weighted_sample(
                context,
                y * context.width + x,
                u,
                v,
                weight,
            )


def polyline_rasterize(
    context: VectorRasterContext,
    points: list[tuple[float, float]],
    params: PolylineShapeParams,
) -> None:
    """stroke or fill a polyline from ordered xz points"""
    if len(points) == 0:
        return
    closed: bool = bool(params.closed.value)
    match context.apply_mode:
        case VectorApplyMode.stroke:
            polyline_stroke_sdf(context, points, closed)
        case VectorApplyMode.fill:
            # open / short polylines cannot form a pad — use the ribbon instead
            if not closed or len(points) < 3:
                polyline_stroke_sdf(context, points, closed)
                return
            pad: float = context.radius_metres
            min_x: float = min(p[0] for p in points) - pad
            max_x: float = max(p[0] for p in points) + pad
            min_z: float = min(p[1] for p in points) - pad
            max_z: float = max(p[1] for p in points) + pad
            ranges: tuple[int, int, int, int] | None = world_aabb_to_texel_range(
                min_x,
                max_x,
                min_z,
                max_z,
                context.size_x,
                context.size_z,
                context.width,
                context.height,
            )
            if ranges is None:
                return
            x0: int
            x1: int
            y0: int
            y1: int
            x0, x1, y0, y1 = ranges
            inv_w: float = 1.0 / float(context.width)
            inv_h: float = 1.0 / float(context.height)
            y: int
            for y in range(y0, y1):
                v: float = (float(y) + 0.5) * inv_h
                wz: float = (v - 0.5) * context.size_z
                x: int
                for x in range(x0, x1):
                    u: float = (float(x) + 0.5) * inv_w
                    wx: float = (u - 0.5) * context.size_x
                    dist: float = distance_to_segments(
                        wx, wz, points, closed=True
                    )
                    inside: bool = point_in_polygon(wx, wz, points)
                    signed: float = -dist if inside else dist
                    weight: float = soft_band_weight(
                        signed, context.radius_metres
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


polyline_shape_spec: VectorShapeSpec = VectorShapeSpec(
    kind="polyline",
    label="Polyline",
    params_type=PolylineShapeParams,
    default_apply_mode=VectorApplyMode.stroke,
    rasterize=polyline_rasterize,
)
