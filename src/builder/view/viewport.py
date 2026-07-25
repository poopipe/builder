"""3D viewport: camera, grid, scene nodes, selection, and gizmo"""

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
    end_mode_3d,
    end_scissor_mode,
    get_mouse_position,
    get_mouse_wheel_move,
    is_key_down,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    rl_disable_depth_test,
    rl_draw_render_batch_active,
    rl_enable_depth_test,
    vector3_add,
    vector3_cross_product,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.meshes.builtins import make_cube
from builder.scene.scene import Scene
from builder.scene.scene_types import builtin_cube, MeshId, Node, Quaternion
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
from builder.view.draw_cache import DrawCache, clear_draw_cache
from builder.view.instances import draw_nodes_instanced
from builder.view.lighting import Lighting
from builder.view.mesh_table import MeshTable
from builder.view.picking import pick_nearest_mesh_node
from builder.view.prepare_mesh import PreparedMesh, prepare_mesh
from builder.view.upload_mesh import upload_imported_mesh
from builder.view.viewport_ray import mouse_in_rect, viewport_world_ray
from builder.view.world_axes import draw_world_axes


@dataclass
class OrbitState:
    """spherical orbit camera around a target (distances in meters)"""

    yaw: float = 45.0
    pitch: float = 30.0
    distance: float = 10.0
    target: Vector3 | None = None

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = Vector3(0.0, 0.0, 0.0)


class Viewport:
    """owns the 3D camera, lighting, prepared meshes, and scene nodes"""

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
        self.draw_cache: DrawCache = DrawCache()
        self.gizmo: GizmoState = GizmoState()
        try:
            self.mesh_table.add(
                builtin_cube,
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
        pitch: float = max(-89.0, min(89.0, self.orbit.pitch))
        self.orbit.pitch = pitch
        yaw_r: float = radians(self.orbit.yaw)
        pitch_r: float = radians(pitch)
        target: Vector3 | None = self.orbit.target
        assert target is not None
        x: float = target.x + self.orbit.distance * cos(pitch_r) * sin(yaw_r)
        y: float = target.y + self.orbit.distance * sin(pitch_r)
        z: float = target.z + self.orbit.distance * cos(pitch_r) * cos(yaw_r)
        self.camera.position = Vector3(x, y, z)
        self.camera.target = target

    def focus_origin(self) -> None:
        """reset the orbit target to the origin"""
        self.orbit.target = Vector3(0.0, 0.0, 0.0)
        self.orbit.distance = 10.0
        self.orbit.yaw = 45.0
        self.orbit.pitch = 30.0
        self.apply_orbit()

    def toggle_grid(self) -> None:
        """toggle ground grid visibility"""
        self.show_grid = not self.show_grid

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """insert or replace scene nodes"""
        self.nodes.add_nodes(nodes)

    def clear_nodes(self) -> None:
        """remove every scene node"""
        self.nodes.clear_nodes()
        end_gizmo_drag(self.gizmo)

    def set_selection(self, group_ids: Sequence[str]) -> None:
        """replace the active group selection"""
        self.nodes.set_selection(group_ids)
        end_gizmo_drag(self.gizmo)

    def clear_selection(self) -> None:
        """clear the active group selection"""
        self.nodes.clear_selection()
        end_gizmo_drag(self.gizmo)

    def register_mesh(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """insert or replace a prepared mesh in the mesh table"""
        self.mesh_table.add_or_replace(mesh_id, prepared)

    def register_imported_mesh(self, imported: ImportedMesh) -> MeshId:
        """upload and register an imported mesh; return its mesh id"""
        mesh_id: MeshId = MeshId(mesh_id_for_import(imported))
        prepared: PreparedMesh = upload_imported_mesh(imported, self.lighting.shader)
        self.register_mesh(mesh_id, prepared)
        return mesh_id

    def clear_non_builtin_meshes(self) -> None:
        """unload every imported mesh; keep built-in cube GPU data"""
        self.mesh_table.clear_except({builtin_cube})

    def rebind_mesh_id(self, from_id: MeshId, to_id: MeshId) -> None:
        """move a prepared mesh to a new id (for stable scene-file ids)"""
        if from_id == to_id:
            return
        if not self.mesh_table.has_mesh(from_id):
            raise KeyError(f"mesh not registered: {from_id.name}")
        prepared: PreparedMesh = self.mesh_table.entries.pop(from_id)
        self.mesh_table.add_or_replace(to_id, prepared)

    def handle_selection_click(self, ray: Ray) -> None:
        """select a group from a mesh hit, or clear on a miss"""
        hit_id: str | None = pick_nearest_mesh_node(
            self.nodes, self.mesh_table, ray, self.draw_cache
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
        """orbit / pan / zoom / select / gizmo when the mouse is over the viewport"""
        mouse: Vector2 = get_mouse_position()
        over: bool = mouse_in_rect(mouse, view_rect)
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
            dx: float = mouse.x - self.last_mouse.x
            dy: float = mouse.y - self.last_mouse.y
            self.orbit.yaw -= dx * 0.35
            self.orbit.pitch += dy * 0.35
            self.last_mouse = mouse
            self.apply_orbit()

        if self.dragging_pan and is_mouse_button_down(MouseButton.MOUSE_BUTTON_MIDDLE):
            dx = mouse.x - self.last_mouse.x
            dy = mouse.y - self.last_mouse.y
            self.last_mouse = mouse
            forward: Vector3 = vector3_normalize(
                vector3_subtract(self.camera.target, self.camera.position)
            )
            right: Vector3 = vector3_normalize(
                vector3_cross_product(forward, self.camera.up)
            )
            up: Vector3 = vector3_normalize(vector3_cross_product(right, forward))
            scale: float = self.orbit.distance * 0.0025
            delta: Vector3 = vector3_add(
                vector3_scale(right, -dx * scale),
                vector3_scale(up, dy * scale),
            )
            assert self.orbit.target is not None
            self.orbit.target = vector3_add(self.orbit.target, delta)
            self.apply_orbit()

        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            self.orbit.distance = max(0.5, min(200.0, self.orbit.distance - wheel * 1.5))
            self.apply_orbit()

    def draw(self, view_rect: Rectangle) -> None:
        """render the 3D scene into the viewport rectangle using scissor"""
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
        draw_nodes_instanced(
            self.mesh_table, self.nodes, self.draw_cache, self.lighting
        )

        # flush scene draws, then submit gizmos with depth off and flush again
        # before restoring depth — otherwise EndMode3D draws them with depth on
        rl_draw_render_batch_active()
        rl_disable_depth_test()
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
        rl_draw_render_batch_active()
        rl_enable_depth_test()

        end_mode_3d()
        end_scissor_mode()

    def unload(self) -> None:
        """release GPU resources and drop large CPU caches before process exit"""
        self.nodes.nodes.clear()
        self.nodes.selected_ids.clear()
        self.nodes.dirty_ids.clear()
        self.nodes.added_ids.clear()
        self.nodes.removed.clear()
        clear_draw_cache(self.draw_cache)
        self.mesh_table.unload()
        self.lighting.unload()
