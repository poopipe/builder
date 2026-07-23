"""Gizmo mode and space enums."""

from __future__ import annotations

from enum import Enum


class GizmoMode(Enum):
    """Active transform gizmo tool."""

    translate = "translate"
    rotate = "rotate"
    scale = "scale"


class GizmoSpace(Enum):
    """Whether gizmo axes follow world or local orientation."""

    world = "world"
    local = "local"


class GizmoAxis(Enum):
    """Gizmo handle axis (or uniform for scale)."""

    x = "x"
    y = "y"
    z = "z"
    uniform = "uniform"
