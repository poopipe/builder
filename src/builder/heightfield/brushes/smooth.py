"""smooth brush — lerp paint samples toward a local neighborhood average"""

from __future__ import annotations

from dataclasses import dataclass

from builder.heightfield.brushes.brush_types import (
    BrushSpec,
    BrushStrokeContext,
    brush_falloff,
    clamp_paint_sample,
    neighborhood_average,
)
from builder.heightfield.sparse_height import (
    SparseHeightRaster,
    get_sample,
    set_sample,
)


@dataclass(frozen=True)
class SmoothBrushParams:
    """kind-specific params for smooth"""


def smooth_stamp(
    raster: SparseHeightRaster,
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: SmoothBrushParams,
) -> None:
    """lerp samples toward 3x3 neighborhood averages under the brush"""
    if width < 1 or height < 1:
        return
    strength: float = max(0.0, min(1.0, float(context.intensity)))
    radius_px_u: float = context.radius_u * float(width)
    radius_px_v: float = context.radius_v * float(height)
    cx: float = context.center_u * float(width)
    cy: float = context.center_v * float(height)
    x0: int = max(0, int(cx - radius_px_u) - 1)
    x1: int = min(width - 1, int(cx + radius_px_u) + 1)
    y0: int = max(0, int(cy - radius_px_v) - 1)
    y1: int = min(height - 1, int(cy + radius_px_v) + 1)
    # snapshot neighborhood reads before writes in this stamp
    reads: dict[tuple[int, int], float] = {}
    y: int
    for y in range(max(0, y0 - 1), min(height, y1 + 2)):
        x: int
        for x in range(max(0, x0 - 1), min(width, x1 + 2)):
            reads[(x, y)] = get_sample(raster, x, y)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            u: float = (float(x) + 0.5) / float(width)
            v: float = (float(y) + 0.5) / float(height)
            w: float = brush_falloff(u, v, context)
            if w <= 0.0:
                continue
            mix: float = w * strength
            if mix <= 0.0:
                continue
            avg: float = neighborhood_average(raster, width, height, x, y)
            # use pre-stamp center value for stable mix
            current: float = reads.get((x, y), 0.0)
            # neighborhood_average reads live raster; rebuild avg from snapshot
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
                    total += reads.get((nx, ny), 0.0)
                    count += 1
            if count > 0:
                avg = total / float(count)
            set_sample(
                raster,
                x,
                y,
                clamp_paint_sample(current + (avg - current) * mix),
            )


smooth_brush_spec: BrushSpec = BrushSpec(
    kind="smooth",
    label="Smooth",
    params_type=SmoothBrushParams,
    stamp=smooth_stamp,
)
