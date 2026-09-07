"""radial generator: stacked rings around a chosen cylinder axis"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Vector3

from builder.distribution.grids import radial_grid_transforms
from builder.generators.generator_types import (
    Axis,
    BuildContext,
    Facing,
    GeneratedSlot,
    GeneratorSpec,
)
from builder.generators.orient import apply_orients, slots_from_transforms
from builder.generators.param_types import (
    EnumParam,
    Float3Param,
    FloatParam,
    IntParam,
)


@dataclass(frozen=True)
class RadialParams:
    """recipe values for a radial generator"""

    axis: EnumParam = EnumParam(Axis.Y, label="Axis")
    radius: FloatParam = FloatParam(5.0, label="Outer radius", minimum=0.1, maximum=50.0)
    spacing: FloatParam = FloatParam(
        30.0, label="Spacing deg", step=5.0, minimum=1.0, maximum=180.0
    )
    count_radius: IntParam = IntParam(1, label="Count radius", minimum=1, maximum=512)
    spacing_radius: FloatParam = FloatParam(
        2.0, label="Spacing radius", minimum=0.1, maximum=50.0
    )
    count_height: IntParam = IntParam(1, label="Count height", minimum=1, maximum=512)
    spacing_height: FloatParam = FloatParam(
        2.0, label="Spacing height", minimum=0.1, maximum=50.0
    )
    facing: EnumParam = EnumParam(
        Facing.center,
        label="Facing",
        options=(Facing.none, Facing.center),
    )
    build_from: EnumParam = EnumParam(Axis.Y, label="Build from")
    orient: Float3Param = Float3Param(
        label="Orient",
        label_x="Yaw",
        label_y="Pitch",
        label_z="Roll",
        step=5.0,
        minimum=-180.0,
        maximum=180.0,
    )


def radial_from_params(params: RadialParams, _: BuildContext) -> list[GeneratedSlot]:
    """build radial slots from typed params"""
    slots: list[GeneratedSlot] = slots_from_transforms(
        radial_grid_transforms(
            Vector3(0.0, 0.0, 0.0),
            radius=params.radius.value,
            spacing=params.spacing.value,
            face_center=params.facing.value is Facing.center,
            axis=int(params.axis.value),
            count_height=params.count_height.value,
            spacing_height=params.spacing_height.value,
            count_radius=params.count_radius.value,
            spacing_radius=params.spacing_radius.value,
            build_from=int(params.build_from.value),
        )
    )
    return apply_orients(
        slots, orient=params.orient, point_orient=params.orient
    )


radial_spec: GeneratorSpec = GeneratorSpec(
    kind="radial",
    label="Radial",
    build_transforms=radial_from_params,
    params_type=RadialParams,
)
