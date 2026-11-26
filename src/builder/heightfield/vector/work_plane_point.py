"""shared work-plane point cubes and xz drag (vector + road vertices)"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import Color, Matrix, Ray, Vector3, draw_cube

from builder.heightfield.vector.circle_handle import ray_hit_hf_plane_xz
from builder.scene.scene_types import Node
from builder.scene.transforms import world_matrix

# match vector overlay point cubes
work_plane_point_color: Color = Color(255, 220, 80, 255)
work_plane_point_selected_color: Color = Color(255, 140, 40, 255)


def draw_work_plane_point(
    x: float,
    plane_y: float,
    z: float,
    cube_size: float,
    *,
    selected: bool,
) -> None:
    """draw one point cube on a work plane"""
    color: Color = (
        work_plane_point_selected_color if selected else work_plane_point_color
    )
    draw_cube(
        Vector3(x, plane_y, z),
        cube_size,
        cube_size,
        cube_size,
        color,
    )


@dataclass
class WorkPlanePointDrag:
    """grab offset so the point does not jump under the cursor"""

    heightfield_group_id: str
    plane_y: float
    grab_dx: float
    grab_dz: float


def begin_work_plane_point_drag(
    ray: Ray,
    hf_world: Matrix,
    plane_y: float,
    point_x: float,
    point_z: float,
    heightfield_group_id: str,
) -> WorkPlanePointDrag | None:
    """start an xz drag from the current ray hit relative to the point"""
    hit_xz: tuple[float, float] | None = ray_hit_hf_plane_xz(
        ray, hf_world, plane_y
    )
    if hit_xz is None:
        return None
    return WorkPlanePointDrag(
        heightfield_group_id=heightfield_group_id,
        plane_y=plane_y,
        grab_dx=point_x - hit_xz[0],
        grab_dz=point_z - hit_xz[1],
    )


def work_plane_point_drag_xz(
    nodes: dict[str, Node],
    drag: WorkPlanePointDrag,
    ray: Ray,
) -> tuple[float, float] | None:
    """xz for the dragged point from the ray on the work plane"""
    hf_world: Matrix = world_matrix(nodes, drag.heightfield_group_id)
    hit_xz: tuple[float, float] | None = ray_hit_hf_plane_xz(
        ray, hf_world, drag.plane_y
    )
    if hit_xz is None:
        return None
    return hit_xz[0] + drag.grab_dx, hit_xz[1] + drag.grab_dz
