""" template for a new generator - copy this file, rename it, then register the spec

to add a generator:
1. copy this file to generators/<your_kind>.py
2. write the transform builder - pure, one Transform per placed mesh
3. describe the editable params in the spec - fields drive the inspector
4. add the spec to GENERATOR_SPECS in generators/registry.py
5. add a place command in commands/scene.py and a panel button in commands/menus.py

simple shared math belongs in distribution/; complex generator-specific math
can live in this file
"""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, radians, sin

from pyray import Transform, Vector3

from builder.generators.generator_types import (
    GeneratorSpec,
    ParamField,
    ParamValue,
)
from builder.scene.scene_types import transform_at


def template_spiral_transforms(
    origin: Vector3,
    count: int,
    turns: float,
    radius: float,
) -> list[Transform]:
    """ return transforms spiralling outward in XZ from origin """
    transforms: list[Transform] = []
    if count <= 0:
        return transforms
    total_degrees: float = 360.0 * turns
    index: int
    for index in range(count):
        # fraction along the spiral; single placement sits at the center
        fraction: float = 0.0 if count == 1 else float(index) / float(count - 1)
        angle: float = radians(total_degrees * fraction)
        distance: float = radius * fraction
        position: Vector3 = Vector3(
            origin.x + cos(angle) * distance,
            origin.y,
            origin.z + sin(angle) * distance,
        )
        transforms.append(transform_at(position))
    return transforms


def template_from_params(params: Mapping[str, ParamValue]) -> list[Transform]:
    """ build template transforms from a param map """
    return template_spiral_transforms(
        Vector3(0.0, 0.0, 0.0),
        count=int(params["count"]),
        turns=float(params["turns"]),
        radius=float(params["radius"]),
    )


TEMPLATE_SPEC: GeneratorSpec = GeneratorSpec(
    kind="template_spiral",
    label="Spiral",
    fields=(
        ParamField("count", "Count", "int", 1.0, minimum=1.0, maximum=512.0),
        ParamField("turns", "Turns", "float", 0.25, minimum=0.1, maximum=32.0),
        ParamField("radius", "Radius", "float", 0.5, minimum=0.1, maximum=100.0),
    ),
    defaults={"count": 24, "turns": 2.0, "radius": 8.0},
    build_transforms=template_from_params,
)
