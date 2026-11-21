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


@dataclass(frozen=True)
class SmoothBrushParams:
    """kind-specific params for smooth"""


def smooth_stamp(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: SmoothBrushParams,
) -> None:
    """lerp samples toward 3x3 neighborhood averages under the brush"""
    if width < 1 or height < 1 or len(samples) != width * height:
        return
    snapshot: list[float] = list(samples)
    strength: float = max(0.0, min(1.0, float(context.intensity)))
    radius_px_u: float = context.radius_u * float(width)
    radius_px_v: float = context.radius_v * float(height)
    cx: float = context.center_u * float(width)
    cy: float = context.center_v * float(height)
    x0: int = max(0, int(cx - radius_px_u) - 1)
    x1: int = min(width - 1, int(cx + radius_px_u) + 1)
    y0: int = max(0, int(cy - radius_px_v) - 1)
    y1: int = min(height - 1, int(cy + radius_px_v) + 1)
    y: int
    for y in range(y0, y1 + 1):
        x: int
        for x in range(x0, x1 + 1):
            u: float = (float(x) + 0.5) / float(width)
            v: float = (float(y) + 0.5) / float(height)
            w: float = brush_falloff(u, v, context)
            if w <= 0.0:
                continue
            mix: float = w * strength
            if mix <= 0.0:
                continue
            index: int = y * width + x
            avg: float = neighborhood_average(snapshot, width, height, x, y)
            current: float = samples[index]
            samples[index] = clamp_paint_sample(current + (avg - current) * mix)


smooth_brush_spec: BrushSpec = BrushSpec(
    kind="smooth",
    label="Smooth",
    params_type=SmoothBrushParams,
    stamp=smooth_stamp,
)
