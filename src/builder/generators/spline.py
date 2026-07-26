"""spline generator: cubic bezier path with stacking and edge facing"""

from __future__ import annotations

from collections.abc import Mapping

from builder.distribution.spline import spline_slots_from_context
from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    GeneratorSpec,
    ParamField,
    ParamMap,
    ParamValue,
    axis_choices,
    edge_orient_fields,
    facing_none_edge,
    orient_defaults,
    point_orient_defaults,
    point_orient_fields,
)


def spline_from_params(
    params: Mapping[str, ParamValue],
    context: BuildContext,
) -> list[GeneratedSlot]:
    """build spline slots from params and control-point children"""
    return spline_slots_from_context(
        context,
        spacing=float(params["spacing"]),
        closed=bool(int(params.get("closed", 0))),
        include_points=bool(int(params.get("include_points", 1))),
        facing=int(params.get("facing", 0)),
        point_facing=int(params.get("point_facing", params.get("facing", 0))),
        axis=int(params.get("axis", 1)),
        count_height=int(params.get("count_height", 1)),
        spacing_height=float(params.get("spacing_height", 2.0)),
        build_from=int(params.get("build_from", 1)),
    )


spline_defaults: ParamMap = {
    "axis": 1,
    "closed": 0,
    "spacing": 1.0,
    "count_height": 1,
    "spacing_height": 2.0,
    "include_points": 1,
    "facing": 2,
    "point_facing": 2,
    "build_from": 1,
    **orient_defaults,
    **point_orient_defaults,
}

spline_spec: GeneratorSpec = GeneratorSpec(
    kind="spline",
    label="Spline",
    fields=(
        ParamField("axis", "Axis", "enum", 1.0, options=axis_choices),
        ParamField("closed", "Closed", "bool", 1.0),
        ParamField(
            "spacing", "Spacing", "float", 0.25, minimum=0.1, maximum=50.0
        ),
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
        ParamField("include_points", "Points", "bool", 1.0),
        ParamField("facing", "Edge facing", "enum", 1.0, options=facing_none_edge),
        ParamField(
            "point_facing", "Point facing", "enum", 1.0, options=facing_none_edge
        ),
        ParamField("build_from", "Build from", "enum", 1.0, options=axis_choices),
        *edge_orient_fields,
        *point_orient_fields,
    ),
    defaults=spline_defaults,
    build_transforms=spline_from_params,
    supports_point_meshes=True,
)
