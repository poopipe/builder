"""generator recipe types attached to group nodes"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pyray import Transform

if TYPE_CHECKING:
    from builder.scene.scene_types import MeshId

type ParamValue = int | float
type ParamMap = dict[str, ParamValue]
type BuildTransforms = Callable[[Mapping[str, ParamValue]], list[Transform]]


@dataclass(frozen=True)
class ParamField:
    """one editable parameter exposed by a generator kind"""

    key: str
    label: str
    value_type: Literal["int", "float"]
    step: float
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class GeneratorSpec:
    """registry entry: how to build and edit one generator kind"""

    kind: str
    label: str
    fields: tuple[ParamField, ...]
    defaults: ParamMap
    build_transforms: BuildTransforms


@dataclass(frozen=True)
class Generator:
    """parametric recipe owned by a group node until baked"""

    kind: str
    mesh_id: MeshId
    params: ParamMap
