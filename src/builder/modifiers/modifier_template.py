"""template for a new modifier - copy this file, rename it, then register the spec

1. copy this file to modifiers/<your_kind>.py
2. declare a frozen params dataclass using IntParam/FloatParam/BoolParam/EnumParam/Float3Param/Int3Param
3. write apply(slots, params, context) reading .value (and Float3Param x/y/z);
   may rewrite transforms or drop slots; must not invent new slots
4. set ModifierSpec(kind, label, params_type, apply)
5. register the spec in registry.modifier_specs; add a panel button
"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Transform, Vector3, vector3_add

from builder.generators.generator_types import (
    Axis,
    BuildContext,
    GeneratedSlot,
)
from builder.generators.param_types import EnumParam, FloatParam
from builder.modifiers.modifier_types import ModifierSpec


@dataclass(frozen=True)
class TemplateTranslateParams:
    """recipe values for the translate-along-axis template modifier"""

    axis: EnumParam = EnumParam(Axis.Y, label="Axis")
    amount: FloatParam = FloatParam(
        1.0, label="Amount", step=0.25, minimum=-50.0, maximum=50.0
    )


def axis_offset(axis: int, amount: float) -> Vector3:
    """return a displacement vector along axis"""
    if axis == 0:
        return Vector3(amount, 0.0, 0.0)
    if axis == 1:
        return Vector3(0.0, amount, 0.0)
    return Vector3(0.0, 0.0, amount)


def template_translate_apply(
    slots: list[GeneratedSlot],
    params: TemplateTranslateParams,
    _: BuildContext,
) -> list[GeneratedSlot]:
    """move each slot by amount along the chosen axis"""
    offset: Vector3 = axis_offset(int(params.axis.value), params.amount.value)
    result: list[GeneratedSlot] = []
    slot: GeneratedSlot
    for slot in slots:
        result.append(
            GeneratedSlot(
                Transform(
                    vector3_add(slot.transform.translation, offset),
                    slot.transform.rotation,
                    slot.transform.scale,
                ),
                slot.role,
            )
        )
    return result


template_translate_spec: ModifierSpec = ModifierSpec(
    kind="template_translate",
    label="Translate (template)",
    params_type=TemplateTranslateParams,
    apply=template_translate_apply,
)
