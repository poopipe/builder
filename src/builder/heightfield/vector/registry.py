"""vector shape kind registry"""

from __future__ import annotations

from typing import Any

from builder.heightfield.vector.circle import circle_shape_spec
from builder.heightfield.vector.polyline import polyline_shape_spec
from builder.heightfield.vector.rectangle import rectangle_shape_spec
from builder.heightfield.vector.vector_types import VectorShapeSpec

vector_shape_specs: tuple[VectorShapeSpec, ...] = (
    polyline_shape_spec,
    circle_shape_spec,
    rectangle_shape_spec,
)


def get_vector_shape_spec(kind: str) -> VectorShapeSpec:
    """return the registry spec for a vector shape kind"""
    spec: VectorShapeSpec
    for spec in vector_shape_specs:
        if spec.kind == kind:
            return spec
    raise KeyError(f"unknown vector shape kind: {kind}")


def known_vector_shape_kind(kind: str) -> bool:
    """true when kind is registered"""
    try:
        get_vector_shape_spec(kind)
    except KeyError:
        return False
    return True


def default_vector_shape_params(kind: str) -> Any:
    """return a fresh default params dataclass for a kind"""
    return get_vector_shape_spec(kind).params_type()
