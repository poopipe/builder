"""3D viewport: camera, grid, scene nodes, selection, and gizmo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, radians, sin

from pyray import (
    Camera3D,
    CameraProjection,
    Color,
    KeyboardKey,
    MouseButton,
    Ray,
    Rectangle,
    Vector2,
    Vector3,
    begin_mode_3d,
    begin_scissor_mode,
    clear_background,
    draw_sphere,
    end_mode_3d,
    end_scissor_mode,
    get_mouse_position,
    get_mouse_wheel_move,
    is_key_down,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    vector3_add,
    vector3_cross_product,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.meshes.builtins import make_cube
from builder.scene.scene import Scene
from builder.scene.scene_types import BUILTIN_CUBE, Node, Quaternion
from builder.scene.selection import root_group_id
from builder.view.gizmo import (
    GizmoState,
    apply_drag_to_scene,
    begin_gizmo_drag,
    draw_gizmo,
    end_gizmo_drag,
    gizmo_axis_directions,
    gizmo_size,
    hit_test_gizmo,
    selection_pivot,
    selection_rotation,
)
from builder.view.gizmo_types import GizmoAxis
from builder.view.grid import draw_ground_grid
from builder.view.instances import draw_nodes_instanced
from builder.view.lighting import Lighting
from builder.view.mesh_table import MeshTable
from builder.view.picking import pick_nearest_mesh_node
from builder.view.prepare_mesh import prepare_mesh
from builder.view.viewport_ray import mouse_in_rect, viewport_world_ray
from builder.view.world_axes import draw_world_axes


@dataclass
class OrbitState:
    """Spherical orbit camera around a target."""

    yaw: float = 45.0
    pitch: float = 30.0
    distance: float = 12.0
    target: Vector3 | None = None

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = Vector3(0.0, 0.5, 0.0)


class Viewport:
    """Owns the 3D camera, lighting, prepared meshes, and scene nodes."""

    def __init__(self) -> None:
        self.orbit: OrbitState = OrbitState()
        self.camera: Camera3D = Camera3D()
        self.camera.up = Vector3(0.0, 1.0, 0.0)
        self.camera.fovy = 60.0
        self.camera.projection = CameraProjection.CAMERA_PERSPECTIVE
        self.apply_orbit()

        self.lighting: Lighting = Lighting()
        self.mesh_table: MeshTable = MeshTable()
        self.nodes: Scene = Scene()
        self.gizmo: GizmoState = GizmoState()
        try:
            self.mesh_table.add(
                BUILTIN_CUBE,
                prepare_mesh(make_cube(1.5), self.lighting.shader),
            )
        except (RuntimeError, ValueError):
            self.mesh_table.unload()
            self.lighting.unload()
            raise

        self.show_grid: bool = True
        self.dragging_orbit: bool = False
        self.dragging_pan: bool = False
        self.last_mouse: Vector2 = Vector2(0.0, 0.0)

    def apply_orbit(self) -> None:
        pitch = max(-89.0, min(89.0, self.orbit.pitch))
        self.orbit.pitch = pitch
        yaw_r = radians(self.orbit.yaw)
        pitch_r = radians(pitch)
        target = self.orbit.target
        assert target is not None
        x = target.x + self.orbit.distance * cos(pitch_r) * sin(yaw_r)
        y = target.y + self.orbit.distance * sin(pitch_r)
        z = target.z + self.orbit.distance * cos(pitch_r) * cos(yaw_r)
        self.camera.position = Vector3(x, y, z)
        self.camera.target = target

    def focus_origin(self) -> None:
        """Reset the orbit target to the origin."""
        self.orbit.target = Vector3(0.0, 0.5, 0.0)
        self.orbit.distance = 12.0
        self.orbit.yaw = 45.0
        self.orbit.pitch = 30.0
        self.apply_orbit()

    def toggle_grid(self) -> None:
        """Toggle ground grid visibility."""
        self.show_grid = not self.show_grid

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """Insert or replace scene nodes."""
        self.nodes.add_nodes(nodes)

    def clear_nodes(self) -> None:
        """Remove every scene node."""
        self.nodes.clear_nodes()
        end_gizmo_drag(self.gizmo)

    def set_selection(self, group_ids: Sequence[str]) -> None:
        """Replace the active group selection."""
        self.nodes.set_selection(group_ids)
        end_gizmo_drag(self.gizmo)

    def clear_selection(self) -> None:
        """Clear the active group selection."""
        self.nodes.clear_selection()
        end_gizmo_drag(self.gizmo)

    def handle_selection_click(self, ray: Ray) -> None:
        """Select a group from a mesh hit, or clear on a miss."""
        hit_id: str | None = pick_nearest_mesh_node(
            self.nodes, self.mesh_table, ray
        )
        if hit_id is None:
            self.clear_selection()
            return
        group_id: str = root_group_id(self.nodes.nodes, hit_id)
        additive: bool = is_key_down(KeyboardKey.KEY_LEFT_CONTROL) or is_key_down(
            KeyboardKey.KEY_RIGHT_CONTROL
        )
        if additive:
            self.nodes.toggle_selection(group_id)
        else:
            self.nodes.set_selection([group_id])
        end_gizmo_drag(self.gizmo)

    def handle_input(self, view_rect: Rectangle, ui_blocks_mouse: bool) -> None:
        """Orbit / pan / zoom / select / gizmo when the mouse is over the viewport."""
        mouse = get_mouse_position()
        over = mouse_in_rect(mouse, view_rect)
        if ui_blocks_mouse or not over:
            self.dragging_orbit = False
            self.dragging_pan = False
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                end_gizmo_drag(self.gizmo)
            return

        ray: Ray = viewport_world_ray(self.camera, mouse)
        pivot: Vector3 | None = selection_pivot(self.nodes, self.nodes.selected_ids)
        axes: dict[GizmoAxis, Vector3] | None = None
        size: float = 1.0
        if pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            axes = gizmo_axis_directions(self.gizmo.space, rotation)
            size = gizmo_size(self.camera, pivot)

        if self.gizmo.drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_drag_to_scene(
                    self.nodes,
                    self.gizmo,
                    ray,
                    self.camera.position,
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                end_gizmo_drag(self.gizmo)
            return

        if (
            is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
            and pivot is not None
            and axes is not None
        ):
            handle: GizmoAxis | None = hit_test_gizmo(
                self.gizmo, ray, pivot, axes, size
            )
            if handle is not None:
                begin_gizmo_drag(
                    self.gizmo,
                    self.nodes,
                    self.nodes.selected_ids,
                    ray,
                    handle,
                    pivot,
                    axes,
                    self.camera.position,
                )
                return

        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
            self.handle_selection_click(ray)
            return

        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_RIGHT):
            self.dragging_orbit = True
            self.last_mouse = mouse
        if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_MIDDLE):
            self.dragging_pan = True
            self.last_mouse = mouse
        if is_mouse_button_released(MouseButton.MOUSE_BUTTON_RIGHT):
            self.dragging_orbit = False
        if is_mouse_button_released(MouseButton.MOUSE_BUTTON_MIDDLE):
            self.dragging_pan = False

        if self.dragging_orbit and is_mouse_button_down(MouseButton.MOUSE_BUTTON_RIGHT):
            dx = mouse.x - self.last_mouse.x
            dy = mouse.y - self.last_mouse.y
            self.orbit.yaw -= dx * 0.35
            self.orbit.pitch += dy * 0.35
            self.last_mouse = mouse
            self.apply_orbit()

        if self.dragging_pan and is_mouse_button_down(MouseButton.MOUSE_BUTTON_MIDDLE):
            dx = mouse.x - self.last_mouse.x
            dy = mouse.y - self.last_mouse.y
            self.last_mouse = mouse
            forward = vector3_normalize(
                vector3_subtract(self.camera.target, self.camera.position)
            )
            right = vector3_normalize(vector3_cross_product(forward, self.camera.up))
            up = vector3_normalize(vector3_cross_product(right, forward))
            scale = self.orbit.distance * 0.0025
            delta = vector3_add(
                vector3_scale(right, -dx * scale),
                vector3_scale(up, dy * scale),
            )
            assert self.orbit.target is not None
            self.orbit.target = vector3_add(self.orbit.target, delta)
            self.apply_orbit()

        wheel = get_mouse_wheel_move()
        if wheel != 0.0:
            self.orbit.distance = max(2.0, min(80.0, self.orbit.distance - wheel * 1.5))
            self.apply_orbit()

    def draw(self, view_rect: Rectangle) -> None:
        """Render the 3D scene into the viewport rectangle using scissor."""
        begin_scissor_mode(
            int(view_rect.x),
            int(view_rect.y),
            int(view_rect.width),
            int(view_rect.height),
        )
        clear_background(Color(24, 26, 30, 255))

        self.lighting.update_view_position(self.camera.position)
        begin_mode_3d(self.camera)
        if self.show_grid:
            draw_ground_grid(40, 1.0)
        draw_world_axes()
        draw_nodes_instanced(self.mesh_table, self.nodes)

        pivot: Vector3 | None = selection_pivot(self.nodes, self.nodes.selected_ids)
        if pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            axes: dict[GizmoAxis, Vector3] = gizmo_axis_directions(
                self.gizmo.space, rotation
            )
            draw_gizmo(
                self.gizmo,
                pivot,
                axes,
                gizmo_size(self.camera, pivot),
            )

        for light in self.lighting.lights:
            draw_sphere(light.position, 0.15, Color(255, 220, 120, 255))
        end_mode_3d()
        end_scissor_mode()

    def unload(self) -> None:
        """Release GPU resources."""
        self.mesh_table.unload()
        self.lighting.unload()
