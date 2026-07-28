"""modifier kind registry"""

from __future__ import annotations

from collections.abc import Mapping

from builder.generators.generator_types import Modifier, ParamField, ParamMap, ParamValue
from builder.modifiers.modifier_types import ModifierSpec
from builder.modifiers.noise_displace import noise_displace_spec

modifier_specs: tuple[ModifierSpec, ...] = (noise_displace_spec,)


def get_modifier_spec(kind: str) -> ModifierSpec:
    """return the registry spec for a modifier kind"""
    spec: ModifierSpec
    for spec in modifier_specs:
        if spec.kind == kind:
            return spec
    raise KeyError(f"unknown modifier kind: {kind}")


def known_modifier_kind(kind: str) -> bool:
    """true when kind is registered"""
    try:
        get_modifier_spec(kind)
        return True
    except KeyError:
        return False


def modifier_params_with_defaults(
    kind: str, params: Mapping[str, ParamValue]
) -> ParamMap:
    """return defaults overlaid with stored params"""
    completed: ParamMap = dict(get_modifier_spec(kind).defaults)
    completed.update(params)
    return completed


def make_modifier(kind: str) -> Modifier:
    """return a new enabled modifier with default params"""
    spec: ModifierSpec = get_modifier_spec(kind)
    return Modifier(kind=kind, params=dict(spec.defaults), enabled=True)


def modifier_field_for(kind: str, key: str) -> ParamField:
    """return one param field from a modifier kind"""
    field: ParamField
    for field in get_modifier_spec(kind).fields:
        if field.key == key:
            return field
    raise KeyError(f"unknown param {key!r} for modifier {kind}")
