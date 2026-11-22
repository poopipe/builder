"""shared path-point recipe types"""

from __future__ import annotations

from dataclasses import dataclass

from builder.generators.param_types import FloatParam


@dataclass(frozen=True)
class PathPointRecipe:
    """per-point authoring data (road width scale, etc.)"""

    width_scale: FloatParam
