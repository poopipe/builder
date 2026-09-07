"""grid generator: meshes on a regular XYZ lattice"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Vector3

from builder.distribution.grids import grid_transforms
from builder.generators.generator_types import (
    Axis,
    BuildContext,
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
class GridParams:
    """recipe values for a grid generator"""

    count_x: IntParam = IntParam(3, label="Count X", minimum=1, maximum=512)
    count_y: IntParam = IntParam(1, label="Count Y", minimum=1, maximum=512)
    count_z: IntParam = IntParam(3, label="Count Z", minimum=1, maximum=512)
    spacing_x: FloatParam = FloatParam(2.0, label="Spacing X", minimum=0.1, maximum=50.0)
    spacing_y: FloatParam = FloatParam(2.0, label="Spacing Y", minimum=0.1, maximum=50.0)
    spacing_z: FloatParam = FloatParam(2.0, label="Spacing Z", minimum=0.1, maximum=50.0)
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


def grid_from_params(params: GridParams, _: BuildContext) -> list[GeneratedSlot]:
    """build grid slots from typed params"""
    slots: list[GeneratedSlot] = slots_from_transforms(
        grid_transforms(
            Vector3(0.0, 0.0, 0.0),
            count_x=params.count_x.value,
            count_y=params.count_y.value,
            count_z=params.count_z.value,
            spacing_x=params.spacing_x.value,
            spacing_y=params.spacing_y.value,
            spacing_z=params.spacing_z.value,
            build_from=int(params.build_from.value),
        )
    )
    return apply_orients(
        slots, orient=params.orient, point_orient=params.orient
    )


grid_spec: GeneratorSpec = GeneratorSpec(
    kind="grid",
    label="Grid",
    build_transforms=grid_from_params,
    params_type=GridParams,
)
