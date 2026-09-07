"""modifier recipe types applied after generator build"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from builder.generators.generator_types import BuildContext, GeneratedSlot


type ApplyModifier = Callable[
    [list[GeneratedSlot], Any, BuildContext], list[GeneratedSlot]
]


@dataclass(frozen=True)
class ModifierSpec:
    """registry entry: how to apply and edit one modifier kind"""

    kind: str
    label: str
    params_type: type
    apply: ApplyModifier
