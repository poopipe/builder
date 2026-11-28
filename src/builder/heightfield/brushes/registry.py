"""brush kind registry"""

from __future__ import annotations

from typing import Any

from builder.heightfield.brushes.add import add_brush_spec
from builder.heightfield.brushes.brush_types import BrushSpec
from builder.heightfield.brushes.flatten import (
    flatten_absolute_brush_spec,
    flatten_brush_spec,
    flatten_down_brush_spec,
    flatten_up_brush_spec,
)
from builder.heightfield.brushes.smooth import smooth_brush_spec
from builder.heightfield.brushes.subtract import subtract_brush_spec

brush_specs: tuple[BrushSpec, ...] = (
    add_brush_spec,
    subtract_brush_spec,
    flatten_brush_spec,
    flatten_up_brush_spec,
    flatten_down_brush_spec,
    flatten_absolute_brush_spec,
    smooth_brush_spec,
)

# brushes offered on vector sculpt shapes (paint-only / no-op kinds omitted)
vector_brush_specs: tuple[BrushSpec, ...] = (
    add_brush_spec,
    flatten_brush_spec,
    flatten_up_brush_spec,
    flatten_down_brush_spec,
    flatten_absolute_brush_spec,
)


def get_brush_spec(kind: str) -> BrushSpec:
    """return the registry spec for a brush kind"""
    spec: BrushSpec
    for spec in brush_specs:
        if spec.kind == kind:
            return spec
    raise KeyError(f"unknown brush kind: {kind}")


def known_brush_kind(kind: str) -> bool:
    """true when kind is registered"""
    try:
        get_brush_spec(kind)
        return True
    except KeyError:
        return False


def default_brush_params(kind: str) -> Any:
    """return a fresh default params dataclass for a kind"""
    return get_brush_spec(kind).params_type()
