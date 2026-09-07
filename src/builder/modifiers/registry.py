"""modifier kind registry"""

from __future__ import annotations

from typing import Any

from builder.generators.generator_types import Modifier
from builder.modifiers.modifier_types import ModifierSpec
from builder.modifiers.modifier_template import template_translate_spec
from builder.modifiers.noise_displace import noise_displace_spec

modifier_specs: tuple[ModifierSpec, ...] = (
    noise_displace_spec,
    template_translate_spec,
)


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


def default_modifier_params(kind: str) -> Any:
    """return a fresh default params dataclass for a kind"""
    return get_modifier_spec(kind).params_type()


def make_modifier(kind: str) -> Modifier:
    """return a new enabled modifier with default params"""
    return Modifier(kind=kind, params=default_modifier_params(kind), enabled=True)
