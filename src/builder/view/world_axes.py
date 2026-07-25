"""world-space axis helpers at the origin"""

from __future__ import annotations

from pyray import Color, Vector3, draw_cylinder_ex


def draw_world_axes(length: float = 2.0, radius: float = 0.04) -> None:
    """draw rgb world axes centered on the origin"""
    origin: Vector3 = Vector3(0.0, 0.0, 0.0)
    sides: int = 12
    draw_cylinder_ex(
        origin,
        Vector3(length, 0.0, 0.0),
        radius,
        radius,
        sides,
        Color(220, 70, 70, 255),
    )
    draw_cylinder_ex(
        origin,
        Vector3(0.0, length, 0.0),
        radius,
        radius,
        sides,
        Color(70, 200, 70, 255),
    )
    draw_cylinder_ex(
        origin,
        Vector3(0.0, 0.0, length),
        radius,
        radius,
        sides,
        Color(70, 120, 220, 255),
    )
