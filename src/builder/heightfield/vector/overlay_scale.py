"""screen-space scale helpers for vector overlays and handles"""

from __future__ import annotations

from math import radians, tan

from pyray import Camera3D

overlay_line_pixels: float = 3.0
shadow_line_pixels: float = 3.0
_min_camera_distance: float = 1e-3
_min_viewport_height_px: float = 1.0


def metres_per_pixel_for_camera(
    camera: Camera3D,
    distance: float,
    viewport_height_px: float,
) -> float:
    """world metres covered by one screen pixel at the given distance"""
    safe_distance: float = max(float(distance), _min_camera_distance)
    safe_height: float = max(float(viewport_height_px), _min_viewport_height_px)
    return (
        2.0 * safe_distance * tan(radians(float(camera.fovy) * 0.5))
    ) / safe_height


def overlay_radius_for_camera(
    camera: Camera3D,
    distance: float,
    viewport_height_px: float,
    pixel_width: float,
) -> float:
    """world-space cylinder radius for a given screen-pixel width"""
    return 0.5 * float(pixel_width) * metres_per_pixel_for_camera(
        camera, distance, viewport_height_px
    )


def point_cube_size_for_line_radius(line_radius: float) -> float:
    """cube edge length so half-extent is twice the line radius"""
    return 4.0 * float(line_radius)
