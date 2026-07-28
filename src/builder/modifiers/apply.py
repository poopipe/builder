"""apply an ordered modifier stack to generated slots"""

from __future__ import annotations

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    Modifier,
)
from builder.modifiers.registry import (
    get_modifier_spec,
    known_modifier_kind,
    modifier_params_with_defaults,
)


def apply_modifier_stack(
    slots: list[GeneratedSlot],
    modifiers: tuple[Modifier, ...],
    context: BuildContext,
) -> list[GeneratedSlot]:
    """fold enabled known modifiers over slots in stack order"""
    current: list[GeneratedSlot] = slots
    modifier: Modifier
    for modifier in modifiers:
        if not modifier.enabled:
            continue
        if not known_modifier_kind(modifier.kind):
            continue
        params = modifier_params_with_defaults(modifier.kind, modifier.params)
        current = get_modifier_spec(modifier.kind).apply(current, params, context)
    return current
