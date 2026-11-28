"""brush recipe types for heightfield paint stamps"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from builder.heightfield.sparse_height import (
    SparseHeightRaster,
    get_sample,
)


@dataclass(frozen=True)
class BrushStrokeContext:
    """one stamp application at a field uv"""

    center_u: float
    center_v: float
    # brush radius in field uv units (0..1 spans the full field)
    radius_u: float
    radius_v: float
    intensity: float
    # world height at stroke press for flatten; unused by add/subtract
    flatten_target: float | None = None


type StampBrush = Callable[
    [SparseHeightRaster, int, int, BrushStrokeContext, Any], None
]


@dataclass(frozen=True)
class BrushSpec:
    """registry entry: how to stamp and edit one brush kind"""

    kind: str
    label: str
    params_type: type
    stamp: StampBrush


def clamp_paint_sample(value: float) -> float:
    """keep paint samples in signed -1..1"""
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def neighborhood_average(
    raster: SparseHeightRaster, width: int, height: int, x: int, y: int
) -> float:
    """3x3 mean of samples around (x, y); out-of-bounds neighbors skipped"""
    total: float = 0.0
    count: int = 0
    dy: int
    for dy in (-1, 0, 1):
        dx: int
        for dx in (-1, 0, 1):
            nx: int = x + dx
            ny: int = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            total += get_sample(raster, nx, ny)
            count += 1
    if count == 0:
        return 0.0
    return total / float(count)


# normalized radius inside which flatten weight stays 1
flatten_falloff_inner: float = 0.65


def brush_normalized_dist(
    u: float,
    v: float,
    context: BrushStrokeContext,
) -> float:
    """ellipse-normalized distance from brush center (1 at rim)"""
    du: float = (
        (u - context.center_u) / context.radius_u
        if context.radius_u > 1e-8
        else 0.0
    )
    dv: float = (
        (v - context.center_v) / context.radius_v
        if context.radius_v > 1e-8
        else 0.0
    )
    return (du * du + dv * dv) ** 0.5


def brush_falloff(
    u: float,
    v: float,
    context: BrushStrokeContext,
) -> float:
    """soft disk falloff 0..1 at field uv (smoothstep on normalized radius)"""
    dist: float = brush_normalized_dist(u, v, context)
    if dist >= 1.0:
        return 0.0
    # smoothstep(0, 1, 1 - dist)
    t: float = 1.0 - dist
    return t * t * (3.0 - 2.0 * t)


def brush_falloff_flatten(
    u: float,
    v: float,
    context: BrushStrokeContext,
) -> float:
    """flatten falloff: weight 1 across an inner plateau, soft only on the rim"""
    dist: float = brush_normalized_dist(u, v, context)
    if dist >= 1.0:
        return 0.0
    inner: float = flatten_falloff_inner
    if dist <= inner:
        return 1.0
    # smoothstep across the outer band inner..1
    span: float = 1.0 - inner
    if span <= 1e-8:
        return 1.0
    t: float = (1.0 - dist) / span
    return t * t * (3.0 - 2.0 * t)
