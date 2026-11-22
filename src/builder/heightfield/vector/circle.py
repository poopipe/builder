"""circle vector shape — center point + radius param"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from builder.generators.param_types import FloatParam
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    path_stroke_weight,
    soft_band_weight,
    world_aabb_to_texel_range,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorRasterContext,
    VectorShapeSpec,
)


@dataclass(frozen=True)
class CircleShapeParams:
    """kind-specific params for circle"""

    radius: FloatParam = FloatParam(
        8.0, label="Radius", step=0.5, minimum=0.1
    )


def circle_rim_local_xz(
    center_x: float, center_z: float, radius: float
) -> tuple[float, float]:
    """rim handle sits on +x from the circle center"""
    return (center_x + float(radius), center_z)


def circle_radius_from_rim(
    center_x: float, center_z: float, rim_x: float, rim_z: float
) -> float:
    """radius from center to a rim xz; clamped to a small minimum"""
    return max(0.1, hypot(rim_x - center_x, rim_z - center_z))


def circle_rasterize(
    context: VectorRasterContext,
    points: list[tuple[float, float]],
    params: CircleShapeParams,
) -> None:
    """stroke rim or fill disk from center point + radius"""
    if len(points) == 0:
        return
    cx: float
    cz: float
    cx, cz = points[0]
    radius: float = max(0.1, float(params.radius.value))
    pad: float = context.radius_metres
    min_x: float = cx - radius - pad
    max_x: float = cx + radius + pad
    min_z: float = cz - radius - pad
    max_z: float = cz + radius + pad
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
                    dist_rim: float = abs(hypot(wx - cx, wz - cz) - radius)
                    weight: float = path_stroke_weight(
                        context, dist_rim, x=wx, z=wz
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
                    dist_center: float = hypot(wx - cx, wz - cz)
                    signed: float = dist_center - radius
                    weight = soft_band_weight(
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


circle_shape_spec: VectorShapeSpec = VectorShapeSpec(
    kind="circle",
    label="Circle",
    params_type=CircleShapeParams,
    default_apply_mode=VectorApplyMode.fill,
    rasterize=circle_rasterize,
)
