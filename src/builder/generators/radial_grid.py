""" radial grid generator: meshes evenly spaced around a circle in XZ """

from __future__ import annotations

from collections.abc import Mapping

from pyray import Transform, Vector3

from builder.distribution.grids import radial_grid_transforms
from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
)


def radial_grid_from_params(params: Mapping[str, ParamValue]) -> list[Transform]:
    """ build radial grid transforms from a param map """
    return radial_grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        radius=float(params["radius"]),
        spacing=float(params["spacing"]),
    )


RADIAL_GRID_SPEC: GeneratorSpec = GeneratorSpec(
    kind="radial_grid",
    label="Radial grid",
    fields=(
        ParamField("radius", "Radius", "float", 0.5, minimum=0.1, maximum=50.0),
        ParamField("spacing", "Spacing deg", "float", 5.0, minimum=1.0, maximum=180.0),
    ),
    defaults={"radius": 5.0, "spacing": 30.0},
    build_transforms=radial_grid_from_params,
)
