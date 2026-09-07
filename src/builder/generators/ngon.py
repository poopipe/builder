"""n-gon generator: stacked regular polygons with linear edge spacing"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Vector3

from builder.distribution.grids import ngon_grid_transforms
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
class NgonParams:
    """recipe values for an n-gon generator"""

    axis: EnumParam = EnumParam(Axis.Y, label="Axis")
    sides: IntParam = IntParam(6, label="Sides", minimum=3, maximum=64)
    radius: FloatParam = FloatParam(5.0, label="Outer radius", minimum=0.1, maximum=50.0)
    aspect: FloatParam = FloatParam(1.0, label="Aspect", step=0.1, minimum=0.1, maximum=10.0)
    spacing: FloatParam = FloatParam(1.0, label="Edge spacing", minimum=0.1, maximum=50.0)
    count_radius: IntParam = IntParam(1, label="Count radius", minimum=1, maximum=512)
    spacing_radius: FloatParam = FloatParam(
        2.0, label="Spacing radius", minimum=0.1, maximum=50.0
    )
    count_height: IntParam = IntParam(1, label="Count height", minimum=1, maximum=512)
    spacing_height: FloatParam = FloatParam(
        2.0, label="Spacing height", minimum=0.1, maximum=50.0
    )
    include_points: BoolParam = BoolParam(True, label="Points")
    facing: EnumParam = EnumParam(Facing.edge, label="Facing")
    point_facing: EnumParam = EnumParam(Facing.center, label="Point facing")
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


def ngon_from_params(params: NgonParams, _: BuildContext) -> list[GeneratedSlot]:
    """build n-gon slots from typed params"""
    slots: list[GeneratedSlot] = ngon_grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        radius=params.radius.value,
        sides=params.sides.value,
        spacing=params.spacing.value,
        facing=int(params.facing.value),
        point_facing=int(params.point_facing.value),
        include_points=params.include_points.value,
        aspect=params.aspect.value,
        axis=int(params.axis.value),
        count_height=params.count_height.value,
        spacing_height=params.spacing_height.value,
        count_radius=params.count_radius.value,
        spacing_radius=params.spacing_radius.value,
        build_from=int(params.build_from.value),
    )
    return apply_orients(
        slots, orient=params.orient, point_orient=params.point_orient
    )


ngon_spec: GeneratorSpec = GeneratorSpec(
    kind="ngon",
    label="N-gon",
    build_transforms=ngon_from_params,
    params_type=NgonParams,
    supports_point_meshes=True,
)
