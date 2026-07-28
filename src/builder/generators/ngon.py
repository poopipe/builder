"""n-gon generator: stacked regular polygons with linear edge spacing"""

from __future__ import annotations

from collections.abc import Mapping

from pyray import Vector3

from builder.distribution.grids import ngon_grid_transforms
from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    GeneratorSpec,
    ParamField,
    ParamMap,
    ParamValue,
    axis_choices,
    distribution_group,
    edge_group,
    edge_orient_fields,
    facing_none_center_edge,
    orient_defaults,
    point_group,
    point_orient_defaults,
    point_orient_fields,
    spacing_group,
)


def ngon_from_params(
    params: Mapping[str, ParamValue],
    _: BuildContext,
) -> list[GeneratedSlot]:
    """build n-gon slots from a param map"""
    return ngon_grid_transforms(
        Vector3(0.0, 0.0, 0.0),
        radius=float(params["radius"]),
        sides=int(params["sides"]),
        spacing=float(params["spacing"]),
        facing=int(params.get("facing", 0)),
        point_facing=int(params.get("point_facing", params.get("facing", 0))),
        include_points=bool(int(params.get("include_points", 1))),
        aspect=float(params.get("aspect", 1.0)),
        axis=int(params.get("axis", 1)),
        count_height=int(params.get("count_height", 1)),
        spacing_height=float(params.get("spacing_height", 2.0)),
        count_radius=int(params.get("count_radius", 1)),
        spacing_radius=float(params.get("spacing_radius", 2.0)),
        build_from=int(params.get("build_from", 1)),
    )


ngon_defaults: ParamMap = {
    "axis": 1,
    "sides": 6,
    "radius": 5.0,
    "aspect": 1.0,
    "spacing": 1.0,
    "count_radius": 1,
    "spacing_radius": 2.0,
    "count_height": 1,
    "spacing_height": 2.0,
    "include_points": 1,
    "facing": 2,
    "point_facing": 1,
    "build_from": 1,
    **orient_defaults,
    **point_orient_defaults,
}

ngon_spec: GeneratorSpec = GeneratorSpec(
    kind="ngon",
    label="N-gon",
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
            "sides",
            "Sides",
            "int",
            1.0,
            minimum=3.0,
            maximum=64.0,
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
            "aspect",
            "Aspect",
            "float",
            0.1,
            minimum=0.1,
            maximum=10.0,
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
        ParamField("include_points", "Points", "bool", 1.0, group=distribution_group),
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
            "Edge spacing",
            "float",
            0.25,
            minimum=0.1,
            maximum=50.0,
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
            options=facing_none_center_edge,
            group=edge_group,
        ),
        *edge_orient_fields,
        ParamField(
            "point_facing",
            "Facing",
            "enum",
            1.0,
            options=facing_none_center_edge,
            group=point_group,
        ),
        *point_orient_fields,
    ),
    defaults=ngon_defaults,
    build_transforms=ngon_from_params,
    supports_point_meshes=True,
)
