"""Viewport ray helpers matched to how the 3D view is drawn."""

from __future__ import annotations

from pyray import (
    Camera3D,
    Ray,
    Rectangle,
    Vector2,
    get_screen_height,
    get_screen_to_world_ray_ex,
    get_screen_width,
)


def viewport_world_ray(camera: Camera3D, mouse: Vector2) -> Ray:
    """ build a world ray using the same full-window projection as begin_mode_3d

    the 3d view is drawn with begin_mode_3d (full framebuffer aspect) and only
    clipped by scissor, so picking must use absolute mouse coords and the
    full screen size — not the client-rect size
    """
    return get_screen_to_world_ray_ex(
        mouse,
        camera,
        get_screen_width(),
        get_screen_height(),
    )


def mouse_in_rect(mouse: Vector2, rect: Rectangle) -> bool:
    """ return true if mouse lies inside rect """
    return (
        rect.x <= mouse.x < rect.x + rect.width
        and rect.y <= mouse.y < rect.y + rect.height
    )
