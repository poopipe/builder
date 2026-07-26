"""radial grid generator: stacked rings around a chosen cylinder axis"""

from __future__ import annotations

from collections.abc import Mapping

from pyray import Transform, Vector3

from builder.distribution.grids import radial_grid_transforms
from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
    axis_choices,
)


def radial_grid_from_params(params: Mapping[str, ParamValue]) -> list[Transform]:
    """build radial grid transforms from a param map"""
    return radial_grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        radius=float(params["radius"]),
        spacing=float(params["spacing"]),
        face_center=bool(int(params["face_center"])),
        axis=int(params.get("axis", 1)),
        count_height=int(params.get("count_height", 1)),
        spacing_height=float(params.get("spacing_height", 2.0)),
        count_radius=int(params.get("count_radius", 1)),
        spacing_radius=float(params.get("spacing_radius", 2.0)),
        build_from=int(params.get("build_from", 1)),
    )


radial_grid_spec: GeneratorSpec = GeneratorSpec(
    kind="radial_grid",
    label="Radial grid",
    fields=(
        ParamField("axis", "Axis", "enum", 1.0, options=axis_choices),
        ParamField("radius", "Outer radius", "float", 0.5, minimum=0.1, maximum=50.0),
        ParamField("spacing", "Spacing deg", "float", 5.0, minimum=1.0, maximum=180.0),
        ParamField(
            "count_height", "Count height", "int", 1.0, minimum=1.0, maximum=512.0
        ),
        ParamField(
            "spacing_height",
            "Spacing height",
            "float",
            0.25,
            minimum=0.1,
            maximum=50.0,
        ),
        ParamField(
            "count_radius", "Count radius", "int", 1.0, minimum=1.0, maximum=512.0
        ),
        ParamField(
            "spacing_radius",
            "Spacing radius",
            "float",
            0.25,
            minimum=0.1,
            maximum=50.0,
        ),
        ParamField("face_center", "Face center", "bool", 1.0),
        ParamField("build_from", "Build from", "enum", 1.0, options=axis_choices),
    ),
    defaults={
        "axis": 1,
        "radius": 5.0,
        "spacing": 30.0,
        "count_radius": 1,
        "spacing_radius": 2.0,
        "count_height": 1,
        "spacing_height": 2.0,
        "face_center": 1,
        "build_from": 1,
    },
    build_transforms=radial_grid_from_params,
)
