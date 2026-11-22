"""circle vector shape — center point + radius param"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from builder.generators.param_types import FloatParam
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    path_stroke_weight,
    soft_band_weight,
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
    match context.apply_mode:
        case VectorApplyMode.stroke:
            pad: float = context.radius_metres
            min_x: float = cx - radius - pad
            max_x: float = cx + radius + pad
            min_z: float = cz - radius - pad
            max_z: float = cz + radius + pad
            y: int
            for y in range(context.height):
                x: int
                for x in range(context.width):
                    u: float = (float(x) + 0.5) / float(context.width)
                    v: float = (float(y) + 0.5) / float(context.height)
                    wx: float = (u - 0.5) * context.size_x
                    wz: float = (v - 0.5) * context.size_z
                    if wx < min_x or wx > max_x or wz < min_z or wz > max_z:
                        continue
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
            pad = context.radius_metres
            min_x = cx - radius - pad
            max_x = cx + radius + pad
            min_z = cz - radius - pad
            max_z = cz + radius + pad
            for y in range(context.height):
                for x in range(context.width):
                    u = (float(x) + 0.5) / float(context.width)
                    v = (float(y) + 0.5) / float(context.height)
                    wx = (u - 0.5) * context.size_x
                    wz = (v - 0.5) * context.size_z
                    if wx < min_x or wx > max_x or wz < min_z or wz > max_z:
                        continue
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
