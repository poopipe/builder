"""flatten brushes — pull paint samples toward stroke-start height"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from builder.heightfield.brushes.brush_types import (
    BrushSpec,
    BrushStrokeContext,
    brush_falloff,
    clamp_paint_sample,
)


class FlattenDirection(IntEnum):
    """which way flatten may move samples"""

    both = 1
    up = 2
    down = 3


@dataclass(frozen=True)
class FlattenBrushParams:
    """kind-specific params for flatten"""


def flatten_stamp_directed(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    direction: FlattenDirection,
) -> None:
    """lerp samples toward flatten_target; optionally only raise or only lower"""
    if context.flatten_target is None:
        return
    if width < 1 or height < 1 or len(samples) != width * height:
        return
    target: float = float(context.flatten_target)
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
            index: int = y * width + x
            current: float = samples[index]
            match direction:
                case FlattenDirection.up:
                    if current >= target:
                        continue
                case FlattenDirection.down:
                    if current <= target:
                        continue
                case FlattenDirection.both:
                    pass
            mix: float = w * strength
            samples[index] = clamp_paint_sample(
                current + (target - current) * mix
            )


def flatten_stamp(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: FlattenBrushParams,
) -> None:
    """lerp samples toward flatten_target by falloff * intensity"""
    flatten_stamp_directed(
        samples, width, height, context, FlattenDirection.both
    )


def flatten_up_stamp(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: FlattenBrushParams,
) -> None:
    """raise samples toward flatten_target only"""
    flatten_stamp_directed(
        samples, width, height, context, FlattenDirection.up
    )


def flatten_down_stamp(
    samples: list[float],
    width: int,
    height: int,
    context: BrushStrokeContext,
    _: FlattenBrushParams,
) -> None:
    """lower samples toward flatten_target only"""
    flatten_stamp_directed(
        samples, width, height, context, FlattenDirection.down
    )


flatten_brush_spec: BrushSpec = BrushSpec(
    kind="flatten",
    label="Flatten",
    params_type=FlattenBrushParams,
    stamp=flatten_stamp,
)

flatten_up_brush_spec: BrushSpec = BrushSpec(
    kind="flatten_up",
    label="Flatten Up",
    params_type=FlattenBrushParams,
    stamp=flatten_up_stamp,
)

flatten_down_brush_spec: BrushSpec = BrushSpec(
    kind="flatten_down",
    label="Flatten Down",
    params_type=FlattenBrushParams,
    stamp=flatten_down_stamp,
)
