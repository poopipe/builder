"""generator recipe types attached to group nodes"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal

from pyray import Transform

if TYPE_CHECKING:
    from builder.scene.scene_types import MeshId, Node

type ParamValue = int | float
type ParamMap = dict[str, ParamValue]
type SlotRole = Literal["default", "point", "edge"]


@dataclass(frozen=True)
class GeneratedSlot:
    """one placed mesh instance from a generator build"""

    transform: Transform
    role: SlotRole = "default"


@dataclass(frozen=True)
class BuildContext:
    """scene access for generators that read control-point children"""

    nodes: Mapping[str, Node]
    group_id: str


@dataclass(frozen=True)
class ParamField:
    """inspector descriptor for one editable value"""

    key: str
    label: str
    value_type: Literal["int", "float", "bool", "enum"]
    step: float
    minimum: float | None = None
    maximum: float | None = None
    options: type[IntEnum] | tuple[IntEnum, ...] | None = None


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class Facing(IntEnum):
    none = 0
    center = 1
    edge = 2


type BuildTransforms = Callable[[Any, BuildContext], list[GeneratedSlot]]


@dataclass(frozen=True)
class GeneratorSpec:
    """registry entry: how to build one generator kind"""

    kind: str
    label: str
    build_transforms: BuildTransforms
    params_type: type
    supports_point_meshes: bool = False


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
    """parametric recipe owned by a group node until baked

    params is a frozen dataclass declared by the generator kind
    """

    kind: str
    meshes: MeshPattern
    params: Any
    point_meshes: MeshPattern | None = None
    modifiers: tuple[Modifier, ...] = ()


@dataclass(frozen=True)
class Modifier:
    """one step in a generator's post-process stack

    may rewrite slot transforms or drop slots; must not invent new ones
    """

    kind: str
    params: ParamMap
    enabled: bool = True
