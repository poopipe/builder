"""template for a new generator - copy this file, rename it, then register the spec

1. copy this file to generators/<your_kind>.py
2. declare a frozen params dataclass using IntParam/FloatParam/BoolParam/EnumParam/Float3Param/Int3Param
3. write build_transforms(params, context) reading .value (and Float3Param x/y/z);
   call apply_orients(...) with Float3Param fields you mean to use as euler degrees
4. set GeneratorSpec(kind, label, build_transforms, params_type)
5. register the spec; add a place command and panel button
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

from pyray import Transform, Vector3

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    GeneratorSpec,
)
from builder.generators.orient import apply_orients, slots_from_transforms
from builder.generators.param_types import Float3Param, FloatParam, IntParam
from builder.scene.scene_types import transform_at


@dataclass(frozen=True)
class TemplateSpiralParams:
    """recipe values for the spiral template generator"""

    count: IntParam = IntParam(24, label="Count", minimum=1, maximum=512)
    turns: FloatParam = FloatParam(2.0, label="Turns", minimum=0.1, maximum=32.0)
    radius: FloatParam = FloatParam(8.0, label="Radius", step=0.5, minimum=0.1, maximum=100.0)
    orient: Float3Param = Float3Param(
        label="Orient",
        label_x="Yaw",
        label_y="Pitch",
        label_z="Roll",
        step=5.0,
        minimum=-180.0,
        maximum=180.0,
    )


def template_spiral_transforms(
    origin: Vector3,
    count: int,
    turns: float,
    radius: float,
) -> list[Transform]:
    """return transforms spiralling outward in XZ from origin"""
    transforms: list[Transform] = []
    if count <= 0:
        return transforms
    total_degrees: float = 360.0 * turns
    index: int
    for index in range(count):
        # fraction along the spiral; single placement sits at the center
        fraction: float = 0.0 if count == 1 else float(index) / float(count - 1)
        angle: float = radians(total_degrees * fraction)
        distance: float = radius * fraction
        position: Vector3 = Vector3(
            origin.x + cos(angle) * distance,
            origin.y,
            origin.z + sin(angle) * distance,
        )
        transforms.append(transform_at(position))
    return transforms


def template_from_params(
    params: TemplateSpiralParams,
    _: BuildContext,
) -> list[GeneratedSlot]:
    """build template slots from typed params"""
    slots: list[GeneratedSlot] = slots_from_transforms(
        template_spiral_transforms(
            Vector3(0.0, 0.0, 0.0),
            count=params.count.value,
            turns=params.turns.value,
            radius=params.radius.value,
        )
    )
    return apply_orients(
        slots, orient=params.orient, point_orient=params.orient
    )


template_spec: GeneratorSpec = GeneratorSpec(
    kind="template_spiral",
    label="Spiral",
    build_transforms=template_from_params,
    params_type=TemplateSpiralParams,
)
