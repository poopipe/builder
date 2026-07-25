"""gizmo mode and space enums"""

from __future__ import annotations

from enum import Enum


class GizmoMode(Enum):
    """active transform gizmo tool"""

    translate = "translate"
    rotate = "rotate"
    scale = "scale"


class GizmoSpace(Enum):
    """whether gizmo axes follow world or local orientation"""

    world = "world"
    local = "local"


class GizmoAxis(Enum):
    """gizmo handle axis (or uniform for scale)"""

    x = "x"
    y = "y"
    z = "z"
    uniform = "uniform"
