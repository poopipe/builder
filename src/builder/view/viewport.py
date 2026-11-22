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
    Shader,
    Vector2,
    Vector3,
    begin_mode_3d,
    begin_scissor_mode,
    clear_background,
    end_mode_3d,
    end_scissor_mode,
    get_frame_time,
    get_mouse_position,
    get_mouse_wheel_move,
    is_key_down,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    rl_disable_depth_test,
    rl_draw_render_batch_active,
    rl_enable_depth_test,
    rl_set_clip_planes,
    vector3_add,
    vector3_cross_product,
    vector3_length,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.generators.regenerate import regenerate_splines_touching
from builder.heightfield.gpu import (
    HeightfieldGpuState,
    create_heightfield_gpu_state,
    draw_heightfields,
    unload_heightfield_gpu,
)
from builder.heightfield.sculpt_stroke import draw_sculpt_brush_preview
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.pick import HeightfieldPickHit, pick_nearest_heightfield
from builder.heightfield.vector.circle_handle import (
    CircleRadiusDrag,
    apply_circle_radius_drag,
    pick_circle_radius_handle,
)
from builder.heightfield.vector.overlay import draw_vector_overlays
from builder.heightfield.vector.pick import (
    VectorPointPickHit,
    pick_nearest_vector_point,
)
from builder.heightfield.vector.rasterize import regenerate_vectors_touching
from builder.heightfield.vector.road_handle import (
    RoadHeightDrag,
    apply_road_height_drag,
    pick_road_height_handle,
)
from builder.io.mesh_import import ImportedMesh, mesh_id_for_import
from builder.meshes.builtins import (
    make_cone,
    make_cube,
    make_cylinder,
    make_sphere,
    make_torus,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import (
    MeshId,
    Node,
    Quaternion,
    builtin_cone,
    builtin_cube,
    builtin_cylinder,
    builtin_mesh_ids,
    builtin_sphere,
    builtin_torus,
)
from builder.scene.selection import root_group_id
from builder.session_journal import transform_payload
from builder.view.draw_cache import DrawCache, clear_draw_cache
from builder.view.gizmo import (
    GizmoState,
    apply_drag_to_scene,
    begin_gizmo_drag,
    draw_gizmo,
    end_gizmo_drag,
    filter_gizmo_axes_for_selection,
    gizmo_axis_directions,
    gizmo_size,
    hit_test_gizmo,
    selection_pivot,
    selection_rotation,
    snap_vector_points_to_plane,
)
from builder.view.gizmo_types import GizmoAxis
from builder.view.grid import draw_ground_grid
from builder.view.instances import draw_nodes_instanced
from builder.view.lighting import Lighting
from builder.view.mesh_table import MeshTable
from builder.view.orbit_move import (
    dolly_move_delta,
    fly_move_delta,
    nudged_move_speed,
    view_distance_default,
    view_distance_near,
)
from builder.view.picking import MeshPickHit, pick_nearest_mesh_node
from builder.view.prepare_mesh import PreparedMesh, prepare_mesh, release_prepared_mesh
from builder.view.upload_mesh import upload_imported_mesh
from builder.view.viewport_ray import mouse_in_rect, viewport_world_ray
from builder.view.world_axes import draw_world_axes


@dataclass
class OrbitState:
    """spherical orbit camera around a target (distances in meters)"""

    yaw: float = 45.0
    pitch: float = 30.0
    distance: float = 10.0
    zoom_speed: float = 1.0
    move_speed: float = 100.0
    view_distance: float = view_distance_default
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
        self.heightfield_gpu: HeightfieldGpuState | None = None
        try:
            builtins: tuple[tuple[MeshId, PreparedMesh], ...] = (
                (builtin_cube, prepare_mesh(make_cube(), self.lighting.shader)),
                (
                    builtin_cylinder,
                    prepare_mesh(make_cylinder(), self.lighting.shader),
                ),
                (builtin_cone, prepare_mesh(make_cone(), self.lighting.shader)),
                (builtin_torus, prepare_mesh(make_torus(), self.lighting.shader)),
                (
                    builtin_sphere,
                    prepare_mesh(make_sphere(), self.lighting.shader),
                ),
            )
            mesh_id: MeshId
            prepared: PreparedMesh
            for mesh_id, prepared in builtins:
                self.mesh_table.add(mesh_id, prepared)
            self.heightfield_gpu = create_heightfield_gpu_state()
        except (RuntimeError, ValueError):
            if self.heightfield_gpu is not None:
                unload_heightfield_gpu(self.heightfield_gpu)
                self.heightfield_gpu = None
            self.mesh_table.unload()
            self.lighting.unload()
            raise

        self.show_grid: bool = True
        self.show_heightfield_wireframe: bool = False
        self.dragging_orbit: bool = False
        self.dragging_pan: bool = False
        self.circle_radius_drag: CircleRadiusDrag | None = None
        self.road_height_drag: RoadHeightDrag | None = None
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

    def toggle_heightfield_wireframe(self) -> None:
        """toggle heightfield solid vs wireframe draw"""
        self.show_heightfield_wireframe = not self.show_heightfield_wireframe

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

    def unload_mesh(self, mesh_id: MeshId) -> None:
        """unload one prepared mesh if present; builtins are ignored"""
        if mesh_id in builtin_mesh_ids:
            return
        prepared: PreparedMesh | None = self.mesh_table.entries.pop(mesh_id, None)
        if prepared is not None:
            release_prepared_mesh(prepared)

    def mesh_shader(self) -> Shader:
        """return the lighting shader used for mesh uploads"""
        return self.lighting.shader

    def register_imported_mesh(self, imported: ImportedMesh) -> MeshId:
        """upload and register an imported mesh; return its mesh id"""
        mesh_id: MeshId = MeshId(mesh_id_for_import(imported))
        prepared: PreparedMesh = upload_imported_mesh(imported, self.lighting.shader)
        self.register_mesh(mesh_id, prepared)
        return mesh_id

    def clear_non_builtin_meshes(self) -> None:
        """unload every imported mesh; keep built-in GPU data"""
        self.mesh_table.clear_except(builtin_mesh_ids)

    def rebind_mesh_id(self, from_id: MeshId, to_id: MeshId) -> None:
        """move a prepared mesh to a new id (for stable scene-file ids)"""
        if from_id == to_id:
            return
        if not self.mesh_table.has_mesh(from_id):
            raise KeyError(f"mesh not registered: {from_id.name}")
        prepared: PreparedMesh = self.mesh_table.entries.pop(from_id)
        self.mesh_table.add_or_replace(to_id, prepared)

    def record_gizmo_commit(self, node_ids: Sequence[str]) -> None:
        """log final local transforms after a gizmo mouse-release commit"""
        if self.nodes.journal is None or not node_ids:
            return
        transforms: dict[str, dict[str, list[float]]] = {}
        node_id: str
        for node_id in node_ids:
            node: Node | None = self.nodes.nodes.get(node_id)
            if node is None:
                continue
            transforms[node_id] = transform_payload(node.transform)
        self.nodes.journal.record(
            "gizmo_commit",
            node_ids=list(node_ids),
            transforms=transforms,
        )

    def handle_selection_click(
        self,
        ray: Ray,
        heightmap_catalog: HeightmapCatalog,
        viewport_height_px: float,
    ) -> None:
        """select a vector point, group from mesh/heightfield hit, or clear"""
        additive: bool = is_key_down(KeyboardKey.KEY_LEFT_CONTROL) or is_key_down(
            KeyboardKey.KEY_RIGHT_CONTROL
        )
        point_hit: VectorPointPickHit | None = pick_nearest_vector_point(
            self.nodes.nodes,
            ray,
            self.camera,
            viewport_height_px,
        )
        mesh_hit: MeshPickHit | None = pick_nearest_mesh_node(
            self.nodes, self.mesh_table, ray, self.draw_cache
        )
        field_hit: HeightfieldPickHit | None = pick_nearest_heightfield(
            self.nodes, heightmap_catalog, ray, self.draw_cache
        )
        # point cubes beat the heightfield volume aabb; only lose to a closer mesh
        hit_id: str | None = None
        select_as_group: bool = True
        if point_hit is not None and (
            mesh_hit is None or point_hit.distance <= mesh_hit.distance
        ):
            hit_id = point_hit.point_id
            select_as_group = False
        else:
            best_distance: float = float("inf")
            if mesh_hit is not None:
                best_distance = mesh_hit.distance
                hit_id = mesh_hit.node_id
                select_as_group = True
            if field_hit is not None and field_hit.distance < best_distance:
                hit_id = field_hit.group_id
                select_as_group = True
        if hit_id is None:
            self.clear_selection()
            return
        selected_id: str = (
            root_group_id(self.nodes.nodes, hit_id) if select_as_group else hit_id
        )
        if additive:
            self.nodes.toggle_selection(selected_id)
        else:
            self.nodes.set_selection([selected_id])
        end_gizmo_drag(self.gizmo)

    def fly_keys_held(self) -> bool:
        """true if any wasd/qe fly key is down"""
        return (
            is_key_down(KeyboardKey.KEY_W)
            or is_key_down(KeyboardKey.KEY_A)
            or is_key_down(KeyboardKey.KEY_S)
            or is_key_down(KeyboardKey.KEY_D)
            or is_key_down(KeyboardKey.KEY_Q)
            or is_key_down(KeyboardKey.KEY_E)
        )

    def apply_keyboard_fly(self, text_focused: bool) -> None:
        """wasd strafe on xz, qe world up/down; skipped while typing"""
        if text_focused:
            return
        move_forward: float = 0.0
        move_right: float = 0.0
        move_up: float = 0.0
        if is_key_down(KeyboardKey.KEY_W):
            move_forward += 1.0
        if is_key_down(KeyboardKey.KEY_S):
            move_forward -= 1.0
        if is_key_down(KeyboardKey.KEY_D):
            move_right += 1.0
        if is_key_down(KeyboardKey.KEY_A):
            move_right -= 1.0
        if is_key_down(KeyboardKey.KEY_E):
            move_up += 1.0
        if is_key_down(KeyboardKey.KEY_Q):
            move_up -= 1.0
        if move_forward == 0.0 and move_right == 0.0 and move_up == 0.0:
            return
        delta: Vector3 = fly_move_delta(
            self.orbit.yaw,
            move_forward,
            move_right,
            move_up,
            self.orbit.distance,
            self.orbit.move_speed,
            get_frame_time(),
        )
        if vector3_length(delta) < 1e-12:
            return
        assert self.orbit.target is not None
        self.orbit.target = vector3_add(self.orbit.target, delta)
        self.apply_orbit()

    def apply_fly_wheel(self, text_focused: bool) -> bool:
        """while flying, wheel nudges move_speed; true if the wheel was used"""
        if text_focused or not self.fly_keys_held():
            return False
        wheel: float = get_mouse_wheel_move()
        if wheel == 0.0:
            return False
        # scroll up = faster, matching + stepper
        self.orbit.move_speed = nudged_move_speed(
            self.orbit.move_speed, 1 if wheel > 0.0 else -1
        )
        return True

    def handle_input(
        self,
        view_rect: Rectangle,
        ui_blocks_mouse: bool,
        heightmap_catalog: HeightmapCatalog,
        text_focused: bool,
        *,
        block_left_click: bool = False,
    ) -> None:
        """orbit / pan / zoom / select / gizmo; keyboard fly when not typing"""
        self.apply_keyboard_fly(text_focused)
        fly_used_wheel: bool = self.apply_fly_wheel(text_focused)
        mouse: Vector2 = get_mouse_position()
        over: bool = mouse_in_rect(mouse, view_rect)
        if ui_blocks_mouse or not over:
            self.dragging_orbit = False
            self.dragging_pan = False
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                dragged: list[str] = (
                    list(self.gizmo.drag.start_transforms.keys())
                    if self.gizmo.drag is not None
                    else []
                )
                radius_shape_ids: list[str] = (
                    [self.circle_radius_drag.shape_id]
                    if self.circle_radius_drag is not None
                    else []
                )
                road_point_ids: list[str] = (
                    [self.road_height_drag.point_id]
                    if self.road_height_drag is not None
                    else []
                )
                if dragged:
                    snap_vector_points_to_plane(self.nodes, dragged)
                end_gizmo_drag(self.gizmo)
                self.circle_radius_drag = None
                self.road_height_drag = None
                if dragged:
                    self.record_gizmo_commit(dragged)
                    regenerate_splines_touching(
                        self.nodes,
                        dragged,
                        heightmap_catalog=heightmap_catalog,
                    )
                    regenerate_vectors_touching(
                        heightmap_catalog, self.nodes.nodes, dragged
                    )
                elif radius_shape_ids:
                    regenerate_vectors_touching(
                        heightmap_catalog, self.nodes.nodes, radius_shape_ids
                    )
                elif road_point_ids:
                    regenerate_vectors_touching(
                        heightmap_catalog, self.nodes.nodes, road_point_ids
                    )
            return

        ray: Ray = viewport_world_ray(self.camera, mouse)
        if self.circle_radius_drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_circle_radius_drag(
                    self.nodes,
                    self.circle_radius_drag,
                    ray,
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                release_radius_ids: list[str] = [
                    self.circle_radius_drag.shape_id
                ]
                self.circle_radius_drag = None
                regenerate_vectors_touching(
                    heightmap_catalog, self.nodes.nodes, release_radius_ids
                )
            return

        if self.road_height_drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_road_height_drag(
                    self.nodes,
                    self.road_height_drag,
                    ray,
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                release_road_ids: list[str] = [
                    self.road_height_drag.point_id
                ]
                self.road_height_drag = None
                regenerate_vectors_touching(
                    heightmap_catalog, self.nodes.nodes, release_road_ids
                )
            return

        pivot: Vector3 | None = selection_pivot(self.nodes, self.nodes.selected_ids)
        axes: dict[GizmoAxis, Vector3] | None = None
        size: float = 1.0
        if pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            axes = filter_gizmo_axes_for_selection(
                gizmo_axis_directions(self.gizmo.space, rotation),
                self.nodes,
                self.nodes.selected_ids,
            )
            size = gizmo_size(self.camera, pivot)

        if not block_left_click and is_mouse_button_pressed(
            MouseButton.MOUSE_BUTTON_LEFT
        ):
            radius_drag: CircleRadiusDrag | None = pick_circle_radius_handle(
                self.nodes.nodes,
                ray,
                self.camera,
                float(view_rect.height),
                list(self.nodes.selected_ids),
            )
            if radius_drag is not None:
                self.circle_radius_drag = radius_drag
                return
            height_drag: RoadHeightDrag | None = pick_road_height_handle(
                heightmap_catalog,
                self.nodes.nodes,
                ray,
                self.camera,
                float(view_rect.height),
                list(self.nodes.selected_ids),
            )
            if height_drag is not None:
                self.road_height_drag = height_drag
                return

        if self.gizmo.drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_drag_to_scene(
                    self.nodes,
                    self.gizmo,
                    ray,
                    self.camera.position,
                )
                snap_vector_points_to_plane(
                    self.nodes, self.nodes.selected_ids
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                dragged: list[str] = list(self.gizmo.drag.start_transforms.keys())
                snap_vector_points_to_plane(self.nodes, dragged)
                end_gizmo_drag(self.gizmo)
                self.record_gizmo_commit(dragged)
                regenerate_splines_touching(
                    self.nodes,
                    dragged,
                    heightmap_catalog=heightmap_catalog,
                )
                regenerate_vectors_touching(
                    heightmap_catalog, self.nodes.nodes, dragged
                )
            return

        if not block_left_click and (
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

        if (
            not block_left_click
            and is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
        ):
            self.handle_selection_click(
                ray, heightmap_catalog, float(view_rect.height)
            )
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

        if not fly_used_wheel:
            wheel: float = get_mouse_wheel_move()
            if wheel != 0.0:
                look: Vector3 = vector3_subtract(
                    self.camera.target, self.camera.position
                )
                dolly: Vector3 = dolly_move_delta(
                    look,
                    wheel,
                    self.orbit.distance,
                    self.orbit.zoom_speed,
                )
                assert self.orbit.target is not None
                self.orbit.target = vector3_add(self.orbit.target, dolly)
                self.apply_orbit()

    def draw(
        self,
        view_rect: Rectangle,
        heightmap_catalog: HeightmapCatalog,
        *,
        sculpt_preview_center: Vector3 | None = None,
        sculpt_preview_radius: float = 0.0,
    ) -> None:
        """render the 3D scene into the viewport rectangle using scissor"""
        begin_scissor_mode(
            int(view_rect.x),
            int(view_rect.y),
            int(view_rect.width),
            int(view_rect.height),
        )
        clear_background(Color(24, 26, 30, 255))

        self.lighting.update_view_position(self.camera.position)
        rl_set_clip_planes(view_distance_near, self.orbit.view_distance)
        begin_mode_3d(self.camera)
        if self.show_grid:
            draw_ground_grid(40, 1.0)
        draw_world_axes()
        draw_nodes_instanced(
            self.mesh_table, self.nodes, self.draw_cache, self.lighting
        )
        if self.heightfield_gpu is not None:
            draw_heightfields(
                self.heightfield_gpu,
                heightmap_catalog,
                self.nodes,
                self.draw_cache,
                self.lighting,
                self.camera.position,
                wireframe=self.show_heightfield_wireframe,
            )
        node: Node
        for node in self.nodes.nodes.values():
            if node.heightfield is not None:
                draw_vector_overlays(
                    heightmap_catalog,
                    self.nodes.nodes,
                    node,
                    self.camera,
                    float(view_rect.height),
                    self.nodes.selected_ids,
                )

        # flush scene draws, then submit gizmos with depth off and flush again
        # before restoring depth — otherwise EndMode3D draws them with depth on
        rl_draw_render_batch_active()
        rl_disable_depth_test()
        if sculpt_preview_center is not None and sculpt_preview_radius > 0.0:
            draw_sculpt_brush_preview(
                sculpt_preview_center, sculpt_preview_radius
            )
        pivot: Vector3 | None = selection_pivot(self.nodes, self.nodes.selected_ids)
        if pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            axes: dict[GizmoAxis, Vector3] = filter_gizmo_axes_for_selection(
                gizmo_axis_directions(self.gizmo.space, rotation),
                self.nodes,
                self.nodes.selected_ids,
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
        if self.heightfield_gpu is not None:
            unload_heightfield_gpu(self.heightfield_gpu)
            self.heightfield_gpu = None
        self.mesh_table.unload()
        self.lighting.unload()
