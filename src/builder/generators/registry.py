"""generator kind registry — specs keyed by kind string

One module per generator kind; add new specs to ``generator_specs``
"""

from __future__ import annotations

from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamMap,
    ParamValue,
)
from builder.generators.horizontal_grid import horizontal_grid_spec
from builder.generators.radial_grid import radial_grid_spec

generator_specs: tuple[GeneratorSpec, ...] = (
    horizontal_grid_spec,
    radial_grid_spec,
)


def get_spec(kind: str) -> GeneratorSpec:
    """return the registry spec for a generator kind"""
    spec: GeneratorSpec
    for spec in generator_specs:
        if spec.kind == kind:
            return spec
    raise KeyError(f"unknown generator kind: {kind}")


def default_params(kind: str) -> ParamMap:
    """return a fresh copy of the default params for a kind"""
    return dict(get_spec(kind).defaults)


def field_for(spec: GeneratorSpec, key: str) -> ParamField:
    """return the param field with the given key"""
    field: ParamField
    for field in spec.fields:
        if field.key == key:
            return field
    raise KeyError(f"unknown param {key!r} for generator {spec.kind}")


def clamp_param(field: ParamField, value: ParamValue) -> ParamValue:
    """clamp and coerce a param value to the field's type and range"""
    number: float = float(value)
    if field.minimum is not None:
        number = max(field.minimum, number)
    if field.maximum is not None:
        number = min(field.maximum, number)
    if field.value_type == "int":
        return int(round(number))
    return number


def format_param(field: ParamField, value: ParamValue) -> str:
    """format a param value for the inspector text field"""
    if field.value_type == "int":
        return str(int(value))
    text: str = f"{float(value):.4g}"
    return text


def parse_param(field: ParamField, text: str) -> ParamValue | None:
    """parse typed inspector text into a param value, or None if invalid"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        if field.value_type == "int":
            return clamp_param(field, int(stripped))
        return clamp_param(field, float(stripped))
    except ValueError:
        return None
