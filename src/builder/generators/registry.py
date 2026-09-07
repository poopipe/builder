"""generator kind registry — specs keyed by kind string

One module per generator kind; add new specs to ``generator_specs``
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
)
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


def option_members(field: ParamField) -> tuple[IntEnum, ...]:
    """return the enum members listed on a field"""
    raw: type[IntEnum] | tuple[IntEnum, ...] | None = field.options
    if raw is None:
        return ()
    if isinstance(raw, type):
        return tuple(raw)
    return raw


def enum_option_values(field: ParamField) -> tuple[int, ...]:
    """return the allowed values for an enum field, empty if it has none"""
    return tuple(int(member) for member in option_members(field))


def clamp_param(field: ParamField, value: ParamValue) -> ParamValue:
    """clamp and coerce a param value to the field's type and range"""
    if field.value_type == "bool":
        return 1 if float(value) >= 0.5 else 0
    if field.value_type == "enum":
        candidate: int = int(round(float(value)))
        values: tuple[int, ...] = enum_option_values(field)
        if candidate in values:
            return candidate
        return values[0] if values else candidate
    number: float = float(value)
    if field.minimum is not None:
        number = max(field.minimum, number)
    if field.maximum is not None:
        number = min(field.maximum, number)
    if field.value_type == "int":
        return int(round(number))
    return number


def enum_option_label(field: ParamField, value: ParamValue) -> str:
    """return the label for an enum value, or its number if unknown"""
    member: IntEnum
    for member in option_members(field):
        if int(member) == int(value):
            return member.name.capitalize()
    return str(int(value))


def format_param(field: ParamField, value: ParamValue) -> str:
    """format a param value for the inspector text field"""
    if field.value_type == "bool":
        return "on" if int(value) else "off"
    if field.value_type == "enum":
        return enum_option_label(field, value)
    if field.value_type == "int":
        return str(int(value))
    text: str = f"{float(value):.4g}"
    return text


def parse_param(field: ParamField, text: str) -> ParamValue | None:
    """parse typed inspector text into a param value, or None if invalid"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    if field.value_type == "bool":
        lowered: str = stripped.lower()
        if lowered in ("1", "true", "on", "yes"):
            return 1
        if lowered in ("0", "false", "off", "no"):
            return 0
        return None
    if field.value_type == "enum":
        return None
    try:
        if field.value_type == "int":
            return clamp_param(field, int(stripped))
        return clamp_param(field, float(stripped))
    except ValueError:
        return None
