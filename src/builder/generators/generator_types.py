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
    """one editable parameter exposed by a generator kind

    an enum field carries ordered (value, label) options shown as radio buttons
    """

    key: str
    label: str
    value_type: Literal["int", "float", "bool", "enum"]
    step: float
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[tuple[int, str], ...] | None = None


axis_choices: tuple[tuple[int, str], ...] = ((0, "X"), (1, "Y"), (2, "Z"))


@dataclass(frozen=True)
class GeneratorSpec:
    """registry entry: how to build and edit one generator kind"""

    kind: str
    label: str
    fields: tuple[ParamField, ...]
    defaults: ParamMap
    build_transforms: BuildTransforms


type MeshSequenceMode = Literal["repeat", "pingpong", "random"]


@dataclass(frozen=True)
class MeshPattern:
    """ordered meshes plus how to sequence them across generated slots

    an empty mesh_ids tuple is tolerated (a deleted mesh may empty a slot);
    consumers fall back to the builtin cube
    """

    mesh_ids: tuple[MeshId, ...]
    mode: MeshSequenceMode = "repeat"
    seed: int = 0


@dataclass(frozen=True)
class Generator:
    """parametric recipe owned by a group node until baked"""

    kind: str
    meshes: MeshPattern
    params: ParamMap
