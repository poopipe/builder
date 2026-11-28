"""shared size for road polyline vertex cubes (draw and pick)"""

from __future__ import annotations

from builder.heightfield.vector.overlay_scale import point_cube_size_for_line_radius

# base edge length in metres; pick floor is half of this
vertex_cube_base_size: float = 2.5
vertex_pick_half_extent: float = vertex_cube_base_size * 0.5


def vertex_cube_draw_size(line_radius: float) -> float:
    """cube edge length: at least the base, larger when the camera is far"""
    return max(
        vertex_cube_base_size,
        point_cube_size_for_line_radius(line_radius),
    )


def vertex_pick_radius(line_radius: float) -> float:
    """xz pick radius matching the drawn cube and segment hit padding"""
    return max(
        vertex_pick_half_extent,
        point_cube_size_for_line_radius(line_radius),
        max(line_radius * 3.0, 1.0),
    )
