"""grid generator: meshes on a regular XYZ lattice"""

from __future__ import annotations

from collections.abc import Mapping

from pyray import Transform, Vector3

from builder.distribution.grids import grid_transforms
from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
    axis_choices,
)


def grid_from_params(params: Mapping[str, ParamValue]) -> list[Transform]:
    """build grid transforms from a param map"""
    return grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        count_x=int(params["count_x"]),
        count_y=int(params["count_y"]),
        count_z=int(params["count_z"]),
        spacing_x=float(params["spacing_x"]),
        spacing_y=float(params["spacing_y"]),
        spacing_z=float(params["spacing_z"]),
        build_from=int(params.get("build_from", 1)),
    )


grid_spec: GeneratorSpec = GeneratorSpec(
    kind="grid",
    label="Grid",
    fields=(
        ParamField("count_x", "Count X", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("count_y", "Count Y", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("count_z", "Count Z", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("spacing_x", "Spacing X", "float", 0.25, minimum=0.1, maximum=50.0),
        ParamField("spacing_y", "Spacing Y", "float", 0.25, minimum=0.1, maximum=50.0),
        ParamField("spacing_z", "Spacing Z", "float", 0.25, minimum=0.1, maximum=50.0),
        ParamField("build_from", "Build from", "enum", 1.0, options=axis_choices),
    ),
    defaults={
        "count_x": 3,
        "count_y": 1,
        "count_z": 3,
        "spacing_x": 2.0,
        "spacing_y": 2.0,
        "spacing_z": 2.0,
        "build_from": 1,
    },
    build_transforms=grid_from_params,
)
