"""generator kind registry — specs keyed by kind string

One module per generator kind; add new specs to ``generator_specs``
"""

from __future__ import annotations

from typing import Any

from builder.generators.generator_types import GeneratorSpec
from builder.generators.grid import grid_spec
from builder.generators.ngon import ngon_spec
from builder.generators.radial import radial_spec
from builder.generators.spline import spline_spec

generator_specs: tuple[GeneratorSpec, ...] = (
    grid_spec,
    radial_spec,
    ngon_spec,
    spline_spec,
)


def get_spec(kind: str) -> GeneratorSpec:
    """return the registry spec for a generator kind"""
    spec: GeneratorSpec
    for spec in generator_specs:
        if spec.kind == kind:
            return spec
    raise KeyError(f"unknown generator kind: {kind}")


def default_params(kind: str) -> Any:
    """return a fresh default params dataclass for a kind"""
    return get_spec(kind).params_type()
