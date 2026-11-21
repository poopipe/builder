"""subtract brush — lower paint samples under the falloff disk"""

from __future__ import annotations

from dataclasses import dataclass

from builder.heightfield.brushes.brush_types import (
    BrushSpec,
    BrushStrokeContext,
    brush_falloff,
    clamp_paint_sample,
)


@dataclass(frozen=True)
class SubtractBrushParams:
    """kind-specific params for subtract"""


def subtract_stamp(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: SubtractBrushParams,
) -> None:
    """subtract intensity * falloff from samples under the brush"""
    if width < 1 or height < 1 or len(samples) != width * height:
        return
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
            index: int = y * width + x
            samples[index] = clamp_paint_sample(
                samples[index] - context.intensity * w
            )


subtract_brush_spec: BrushSpec = BrushSpec(
    kind="subtract",
    label="Subtract",
    params_type=SubtractBrushParams,
    stamp=subtract_stamp,
)
