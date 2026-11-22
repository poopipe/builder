"""rectangle vector shape — two opposite corner points"""

from __future__ import annotations

from dataclasses import dataclass

from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    distance_to_segments,
    path_stroke_weight,
    soft_band_weight,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorRasterContext,
    VectorShapeSpec,
)


@dataclass(frozen=True)
class RectangleShapeParams:
    """kind-specific params for rectangle (geometry from two corners)"""


def rectangle_corners(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """axis-aligned corners from two opposite points"""
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


def rectangle_rasterize(
    context: VectorRasterContext,
    points: list[tuple[float, float]],
    _: RectangleShapeParams,
) -> None:
    """stroke perimeter or fill aabb from two corner points"""
    corners: list[tuple[float, float]] | None = rectangle_corners(points)
    if corners is None:
        return
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    min_x, min_z = corners[0]
    max_x, max_z = corners[2]
    match context.apply_mode:
        case VectorApplyMode.stroke:
            pad: float = context.radius_metres
            y: int
            for y in range(context.height):
                x: int
                for x in range(context.width):
                    u: float = (float(x) + 0.5) / float(context.width)
                    v: float = (float(y) + 0.5) / float(context.height)
                    wx: float = (u - 0.5) * context.size_x
                    wz: float = (v - 0.5) * context.size_z
                    if (
                        wx < min_x - pad
                        or wx > max_x + pad
                        or wz < min_z - pad
                        or wz > max_z + pad
                    ):
                        continue
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
            pad = context.radius_metres
            for y in range(context.height):
                for x in range(context.width):
                    u = (float(x) + 0.5) / float(context.width)
                    v = (float(y) + 0.5) / float(context.height)
                    wx = (u - 0.5) * context.size_x
                    wz = (v - 0.5) * context.size_z
                    if (
                        wx < min_x - pad
                        or wx > max_x + pad
                        or wz < min_z - pad
                        or wz > max_z + pad
                    ):
                        continue
                    dx: float
                    if wx < min_x:
                        dx = min_x - wx
                    elif wx > max_x:
                        dx = wx - max_x
                    else:
                        dx = 0.0
                    dz: float
                    if wz < min_z:
                        dz = min_z - wz
                    elif wz > max_z:
                        dz = wz - max_z
                    else:
                        dz = 0.0
                    if dx == 0.0 and dz == 0.0:
                        # inside: distance to nearest edge (negative)
                        signed: float = -min(
                            wx - min_x,
                            max_x - wx,
                            wz - min_z,
                            max_z - wz,
                        )
                    else:
                        signed = (dx * dx + dz * dz) ** 0.5
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
