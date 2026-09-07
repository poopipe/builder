"""generator recipe types attached to group nodes"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyray import Transform

if TYPE_CHECKING:
    from builder.scene.scene_types import MeshId, Node


class SlotRole(IntEnum):
    default = 0
    point = 1
    edge = 2


@dataclass(frozen=True)
class GeneratedSlot:
    """slot for future mesh placement"""

    transform: Transform
    role: SlotRole = SlotRole.default


@dataclass(frozen=True)
class BuildContext:
    """scene access for generators that read control-point children"""

    nodes: Mapping[str, Node]
    group_id: str


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


class MeshSequenceMode(IntEnum):
    repeat = 0
    pingpong = 1
    random = 2


@dataclass(frozen=True)
class MeshPattern:
    """ordered meshes plus how to sequence them across generated slots

    an empty mesh_ids tuple is tolerated (a deleted mesh may empty a slot);
    consumers fall back to the builtin cube
    """

    mesh_ids: tuple[MeshId, ...]
    mode: MeshSequenceMode = MeshSequenceMode.repeat
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

    params is a frozen dataclass declared by the modifier kind;
    may rewrite slot transforms or drop slots; must not invent new ones
    """

    kind: str
    params: Any
    enabled: bool = True
