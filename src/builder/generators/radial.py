"""radial generator: stacked rings around a chosen cylinder axis"""

from __future__ import annotations

from collections.abc import Mapping

from pyray import Vector3

from builder.distribution.grids import radial_grid_transforms
from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    GeneratorSpec,
    ParamField,
    ParamMap,
    ParamValue,
    axis_choices,
    distribution_group,
    facing_none_center,
    orient_defaults,
    orient_fields,
    orientation_group,
    spacing_group,
)
from builder.generators.orient import slots_from_transforms


def radial_facing(params: Mapping[str, ParamValue]) -> bool:
    """resolve facing, migrating legacy face_center when needed"""
    if "facing" in params:
        return int(params["facing"]) == 1
    return bool(int(params.get("face_center", 0)))


def radial_from_params(
    params: Mapping[str, ParamValue],
    _: BuildContext,
) -> list[GeneratedSlot]:
    """build radial slots from a param map"""
    return slots_from_transforms(
        radial_grid_transforms(
            Vector3(0.0, 0.0, 0.0),
            radius=float(params["radius"]),
            spacing=float(params["spacing"]),
            face_center=radial_facing(params),
            axis=int(params.get("axis", 1)),
            count_height=int(params.get("count_height", 1)),
            spacing_height=float(params.get("spacing_height", 2.0)),
            count_radius=int(params.get("count_radius", 1)),
            spacing_radius=float(params.get("spacing_radius", 2.0)),
            build_from=int(params.get("build_from", 1)),
        )
    )


radial_defaults: ParamMap = {
    "axis": 1,
    "radius": 5.0,
    "spacing": 30.0,
    "count_radius": 1,
    "spacing_radius": 2.0,
    "count_height": 1,
    "spacing_height": 2.0,
    "facing": 1,
    "build_from": 1,
    **orient_defaults,
}

radial_spec: GeneratorSpec = GeneratorSpec(
    kind="radial",
    label="Radial",
    fields=(
        ParamField(
            "axis",
            "Axis",
            "enum",
            1.0,
            options=axis_choices,
            group=distribution_group,
        ),
        ParamField(
            "radius",
            "Outer radius",
            "float",
            0.5,
            minimum=0.1,
            maximum=50.0,
            group=distribution_group,
        ),
        ParamField(
            "count_height",
            "Count height",
            "int",
            1.0,
            minimum=1.0,
            maximum=512.0,
            group=distribution_group,
        ),
        ParamField(
            "count_radius",
            "Count radius",
            "int",
            1.0,
            minimum=1.0,
            maximum=512.0,
            group=distribution_group,
        ),
        ParamField(
            "build_from",
            "Build from",
            "enum",
            1.0,
            options=axis_choices,
            group=distribution_group,
        ),
        ParamField(
            "spacing",
            "Spacing deg",
            "float",
            5.0,
            minimum=1.0,
            maximum=180.0,
            group=spacing_group,
        ),
        ParamField(
            "spacing_height",
            "Spacing height",
            "float",
            0.25,
            minimum=0.1,
            maximum=50.0,
            group=spacing_group,
        ),
        ParamField(
            "spacing_radius",
            "Spacing radius",
            "float",
            0.25,
            minimum=0.1,
            maximum=50.0,
            group=spacing_group,
        ),
        ParamField(
            "facing",
            "Facing",
            "enum",
            1.0,
            options=facing_none_center,
            group=orientation_group,
        ),
        *orient_fields,
    ),
    defaults=radial_defaults,
    build_transforms=radial_from_params,
)
