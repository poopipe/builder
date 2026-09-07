"""spline generator: cubic bezier path with stacking and edge facing"""

from __future__ import annotations

from dataclasses import dataclass

from builder.distribution.spline import spline_slots_from_context
from builder.generators.generator_types import (
    Axis,
    BuildContext,
    Facing,
    GeneratedSlot,
    GeneratorSpec,
)
from builder.generators.orient import apply_orients
from builder.generators.param_types import (
    BoolParam,
    EnumParam,
    Float3Param,
    FloatParam,
    IntParam,
)


@dataclass(frozen=True)
class SplineParams:
    """recipe values for a spline generator"""

    axis: EnumParam = EnumParam(Axis.Y, label="Axis")
    closed: BoolParam = BoolParam(False, label="Closed")
    spacing: FloatParam = FloatParam(1.0, label="Spacing", minimum=0.1, maximum=50.0)
    count_height: IntParam = IntParam(1, label="Count height", minimum=1, maximum=512)
    spacing_height: FloatParam = FloatParam(
        2.0, label="Spacing height", minimum=0.1, maximum=50.0
    )
    include_points: BoolParam = BoolParam(True, label="Points")
    facing: EnumParam = EnumParam(
        Facing.edge,
        label="Facing",
        options=(Facing.none, Facing.edge),
    )
    point_facing: EnumParam = EnumParam(
        Facing.edge,
        label="Point facing",
        options=(Facing.none, Facing.edge),
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
    point_orient: Float3Param = Float3Param(
        label="Point orient",
        label_x="Yaw",
        label_y="Pitch",
        label_z="Roll",
        step=5.0,
        minimum=-180.0,
        maximum=180.0,
    )


def spline_from_params(
    params: SplineParams, context: BuildContext
) -> list[GeneratedSlot]:
    """build spline slots from typed params and control-point children"""
    slots: list[GeneratedSlot] = spline_slots_from_context(
        context,
        spacing=params.spacing.value,
        closed=params.closed.value,
        include_points=params.include_points.value,
        facing=int(params.facing.value),
        point_facing=int(params.point_facing.value),
        axis=int(params.axis.value),
        count_height=params.count_height.value,
        spacing_height=params.spacing_height.value,
        build_from=int(params.build_from.value),
    )
    return apply_orients(
        slots, orient=params.orient, point_orient=params.point_orient
    )


spline_spec: GeneratorSpec = GeneratorSpec(
    kind="spline",
    label="Spline",
    build_transforms=spline_from_params,
    params_type=SplineParams,
    supports_point_meshes=True,
)
