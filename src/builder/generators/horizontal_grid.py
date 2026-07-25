""" horizontal grid generator: meshes on a regular XZ lattice """

from __future__ import annotations

from collections.abc import Mapping

from pyray import Transform, Vector3

from builder.distribution.grids import horizontal_grid_transforms
from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
)


def horizontal_grid_from_params(params: Mapping[str, ParamValue]) -> list[Transform]:
    """ build horizontal grid transforms from a param map """
    return horizontal_grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        count_x=int(params["count_x"]),
        count_z=int(params["count_z"]),
        spacing=float(params["spacing"]),
    )


HORIZONTAL_GRID_SPEC: GeneratorSpec = GeneratorSpec(
    kind="horizontal_grid",
    label="Cube grid",
    fields=(
        ParamField("count_x", "Count X", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("count_z", "Count Z", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("spacing", "Spacing", "float", 0.25, minimum=0.1, maximum=50.0),
    ),
    defaults={"count_x": 3, "count_z": 3, "spacing": 2.0},
    build_transforms=horizontal_grid_from_params,
)
