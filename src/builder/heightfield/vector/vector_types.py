"""vector layer / shape recipe types for heightfield edit stacks"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from builder.generators.param_types import FloatParam

# u,v -> world height below the vector sculpt layer
BelowAtUv = Callable[[float, float], float]
# x,z heightfield-local -> scalar (half-width metres)
RibbonScalarAtXz = Callable[[float, float], float]
# x,z,below -> absolute ribbon target height (metres); below avoids a second sample
RibbonTargetAtXz = Callable[[float, float, float], float]


class VectorApplyMode(IntEnum):
    """how a shape stamps its brush into the derived height buffer"""

    stroke = 1
    fill = 2


@dataclass(frozen=True)
class VectorLayerRecipe:
    """marker recipe on a vector layer group under a heightfield"""


@dataclass(frozen=True)
class VectorShapeRecipe:
    """one live vector shape; geometry comes from point children + params"""

    kind: str
    apply_mode: VectorApplyMode
    brush_kind: str
    offset: FloatParam
    intensity: FloatParam
    radius: FloatParam
    # 0 = drape (surface + offset), 1 = grade (absolute knot ribbon)
    grade: FloatParam
    params: object
    enabled: bool = True


@dataclass(frozen=True)
class VectorRasterContext:
    """shared stamp target for one shape rasterize call"""

    samples: list[float]
    width: int
    height: int
    size_x: float
    size_z: float
    radius_metres: float
    # signed add / paint amount
    offset: float
    # flatten blend mix 0..1
    intensity: float
    apply_mode: VectorApplyMode
    brush_kind: str
    # sculpt metre path (paint when false)
    sculpt_metres: bool = False
    flatten_target: float | None = None
    below_at_uv: BelowAtUv | None = None
    # when set, weighted samples flatten toward this absolute height
    ribbon_height_at_xz: RibbonTargetAtXz | None = None
    # when set, road stroke uses per-sample half-width (metres)
    ribbon_half_width_at_xz: RibbonScalarAtXz | None = None
    # road coverage 0..1 (same size as samples); None for non-road rasters
    coverage_samples: list[float] | None = None


type RasterizeShape = Callable[
    [VectorRasterContext, list[tuple[float, float]], Any], None
]


@dataclass(frozen=True)
class VectorShapeSpec:
    """registry entry for one authorable vector shape kind"""

    kind: str
    label: str
    params_type: type
    default_apply_mode: VectorApplyMode
    rasterize: RasterizeShape


def world_xz_to_uv(
    x: float, z: float, size_x: float, size_z: float
) -> tuple[float, float]:
    """map heightfield-local xz (origin-centered) to field uv"""
    u: float = (x + 0.5 * size_x) / size_x if size_x > 1e-8 else 0.5
    v: float = (z + 0.5 * size_z) / size_z if size_z > 1e-8 else 0.5
    return (u, v)


def radius_metres_to_uv(
    radius_metres: float, size_x: float, size_z: float
) -> tuple[float, float]:
    """convert world brush radius to field uv radii"""
    ru: float = radius_metres / size_x if size_x > 1e-8 else 0.0
    rv: float = radius_metres / size_z if size_z > 1e-8 else 0.0
    return (max(ru, 1e-6), max(rv, 1e-6))
