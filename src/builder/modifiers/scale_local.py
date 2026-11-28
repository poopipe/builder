"""multiply each slot's local scale axes independently"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Transform, Vector3

from builder.generators.generator_types import BuildContext, GeneratedSlot
from builder.generators.param_types import Float3Param
from builder.modifiers.modifier_types import ModifierSpec


@dataclass(frozen=True)
class ScaleLocalParams:
    """per-axis local scale factors"""

    scale: Float3Param = Float3Param(
        x=1.0,
        y=1.0,
        z=1.0,
        label="Scale",
        label_x="X",
        label_y="Y",
        label_z="Z",
        step=0.05,
        minimum=0.01,
        maximum=100.0,
    )


def scale_local_apply(
    slots: list[GeneratedSlot],
    params: ScaleLocalParams,
    _: BuildContext,
) -> list[GeneratedSlot]:
    """scale each slot in local space; axes are independent"""
    sx: float = float(params.scale.x)
    sy: float = float(params.scale.y)
    sz: float = float(params.scale.z)
    result: list[GeneratedSlot] = []
    slot: GeneratedSlot
    for slot in slots:
        result.append(
            GeneratedSlot(
                Transform(
                    slot.transform.translation,
                    slot.transform.rotation,
                    Vector3(
                        slot.transform.scale.x * sx,
                        slot.transform.scale.y * sy,
                        slot.transform.scale.z * sz,
                    ),
                ),
                slot.role,
            )
        )
    return result


scale_local_spec: ModifierSpec = ModifierSpec(
    kind="scale_local",
    label="Scale local",
    params_type=ScaleLocalParams,
    apply=scale_local_apply,
)
