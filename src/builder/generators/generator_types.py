"""generator recipe types attached to group nodes"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


type BuildTransforms = Callable[
    [Mapping[str, ParamValue], BuildContext], list[GeneratedSlot]
]


@dataclass(frozen=True)
class ParamField:
    """one editable parameter exposed by a generator kind

    an enum field carries ordered (value, label) options shown as radio buttons

    group names a collapsible section in the inspector; contiguous fields
    sharing a group fold together, empty group stays pinned at the top
    """

    key: str
    label: str
    value_type: Literal["int", "float", "bool", "enum"]
    step: float
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[tuple[int, str], ...] | None = None
    group: str = ""


axis_choices: tuple[tuple[int, str], ...] = ((0, "X"), (1, "Y"), (2, "Z"))

facing_none_center: tuple[tuple[int, str], ...] = (
    (0, "None"),
    (1, "Center"),
)
facing_none_center_edge: tuple[tuple[int, str], ...] = (
    (0, "None"),
    (1, "Center"),
    (2, "Edge"),
)
facing_none_edge: tuple[tuple[int, str], ...] = (
    (0, "None"),
    (2, "Edge"),
)

distribution_group: str = "Distribution"
spacing_group: str = "Spacing"
orientation_group: str = "Orientation"
edge_group: str = "Edge"
point_group: str = "Point"

orient_fields: tuple[ParamField, ...] = (
    ParamField(
        "orient_yaw",
        "Yaw",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=orientation_group,
    ),
    ParamField(
        "orient_pitch",
        "Pitch",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=orientation_group,
    ),
    ParamField(
        "orient_roll",
        "Roll",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=orientation_group,
    ),
)
orient_defaults: ParamMap = {
    "orient_yaw": 0.0,
    "orient_pitch": 0.0,
    "orient_roll": 0.0,
}

# same keys as orient_fields; labels for generators with separate point meshes
edge_orient_fields: tuple[ParamField, ...] = (
    ParamField(
        "orient_yaw",
        "Yaw",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=edge_group,
    ),
    ParamField(
        "orient_pitch",
        "Pitch",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=edge_group,
    ),
    ParamField(
        "orient_roll",
        "Roll",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=edge_group,
    ),
)
point_orient_fields: tuple[ParamField, ...] = (
    ParamField(
        "point_orient_yaw",
        "Yaw",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=point_group,
    ),
    ParamField(
        "point_orient_pitch",
        "Pitch",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=point_group,
    ),
    ParamField(
        "point_orient_roll",
        "Roll",
        "float",
        5.0,
        minimum=-180.0,
        maximum=180.0,
        group=point_group,
    ),
)
point_orient_defaults: ParamMap = {
    "point_orient_yaw": 0.0,
    "point_orient_pitch": 0.0,
    "point_orient_roll": 0.0,
}


@dataclass(frozen=True)
class GeneratorSpec:
    """registry entry: how to build and edit one generator kind"""

    kind: str
    label: str
    fields: tuple[ParamField, ...]
    defaults: ParamMap
    build_transforms: BuildTransforms
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
    """parametric recipe owned by a group node until baked"""

    kind: str
    meshes: MeshPattern
    params: ParamMap
    point_meshes: MeshPattern | None = None
