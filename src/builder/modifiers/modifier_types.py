"""modifier recipe types applied after generator build"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    ParamField,
    ParamMap,
)


type ApplyModifier = Callable[
    [list[GeneratedSlot], ParamMap, BuildContext], list[GeneratedSlot]
]


@dataclass(frozen=True)
class ModifierSpec:
    """registry entry: how to apply and edit one modifier kind"""

    kind: str
    label: str
    fields: tuple[ParamField, ...]
    defaults: ParamMap
    apply: ApplyModifier
