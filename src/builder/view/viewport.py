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
    Matrix,
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
from builder.heightfield.paint import tick_deferred_gpu_flush
from builder.heightfield.pick import HeightfieldPickHit, pick_nearest_heightfield
from builder.heightfield.vector.circle_handle import (
    CircleRadiusDrag,
    apply_circle_radius_drag,
    pick_circle_radius_handle,
)
from builder.heightfield.road_sections.overlay import draw_road_sections_overlays
from builder.heightfield.road_sections.part_catalog import RoadPartsCatalog
from builder.heightfield.road_sections.pick import (
    RoadPolylinePickHit,
    pick_nearest_road_polyline,
)
from builder.heightfield.road_sections.edit import set_usage_group_on_node
from builder.heightfield.road_sections.place import (
    drop_placements,
    regenerate_road_piece_children,
)
from builder.heightfield.road_sections.rasterize import (
    dirty_aabb_for_road_endpoints,
)
from builder.heightfield.road_sections.road_sections_types import RoadEndpointRef
from builder.heightfield.road_sections.vertex_drag import (
    RoadVertexDrag,
    apply_road_vertex_drag,
    pick_road_vertex_drag,
)
from builder.heightfield.road_sections.vertex_gizmo import (
    RoadVertexAxisDrag,
    apply_road_vertex_axis_drag,
    dragged_polyline_ids,
    draw_road_vertex_move_gizmo,
    hf_local_xz_axes,
    pick_road_vertex_axis_drag,
    selection_pivot_world,
)
from builder.heightfield.vector.overlay import draw_vector_overlays
from builder.heightfield.vector.pick import (
    VectorPointPickHit,
    pick_nearest_vector_point,
)
from builder.ui.ui_state import UiState
from builder.heightfield.vector.rasterize import (
    dirty_aabb_for_vector_nodes,
    regenerate_vectors_touching,
    union_aabb,
)
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
    selection_is_vector_workplane_nodes,
    selection_pivot,
    selection_rotation,
    snap_vector_points_to_plane,
    vector_shape_rotate_pivot_and_axes,
)
from builder.view.gizmo_types import GizmoAxis, GizmoMode
from builder.view.grid import draw_ground_grid
from builder.view.instances import draw_nodes_instanced
from builder.view.lighting import Lighting
from builder.view.line_shader import (
    LineBatch,
    LineMeshCache,
    LineShader,
    clear_line_mesh_cache,
    load_line_shader,
    unload_line_shader,
)
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
        self.line_shader: LineShader | None = None
        self.line_batch: LineBatch = LineBatch()
        self.road_line_mesh_cache: LineMeshCache = LineMeshCache()
        self.mesh_table: MeshTable = MeshTable()
        self.nodes: Scene = Scene()
        self.draw_cache: DrawCache = DrawCache()
        self.gizmo: GizmoState = GizmoState()
        self.heightfield_gpu: HeightfieldGpuState | None = None
        try:
            self.line_shader = load_line_shader()
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
            if self.line_shader is not None:
                unload_line_shader(self.line_shader)
                self.line_shader = None
            self.lighting.unload()
            raise

        self.show_grid: bool = True
        self.show_heightfield_wireframe: bool = False
        self.show_sea_plane: bool = True
        self.dragging_orbit: bool = False
        self.dragging_pan: bool = False
        self.circle_radius_drag: CircleRadiusDrag | None = None
        self.road_height_drag: RoadHeightDrag | None = None
        self.road_vertex_drag: RoadVertexDrag | None = None
        self.road_vertex_axis_drag: RoadVertexAxisDrag | None = None
        # stamp footprint before a vector edit (unioned with post-edit dirty)
        self.vector_edit_prior_dirty: (
            tuple[float, float, float, float] | None
        ) = None
        self.road_vertex_prior_dirty: (
            tuple[float, float, float, float] | None
        ) = None
        self.last_mouse: Vector2 = Vector2(0.0, 0.0)

    def begin_vector_edit_dirty(self, node_ids: list[str]) -> None:
        """snapshot shape stamp aabbs before a vector transform/radius edit"""
        self.vector_edit_prior_dirty = dirty_aabb_for_vector_nodes(
            self.nodes.nodes, node_ids
        )

    def begin_road_vertex_edit_dirty(
        self,
        usage_group_id: str,
        endpoints: Sequence[RoadEndpointRef],
        road_parts: RoadPartsCatalog | None,
    ) -> None:
        """snapshot ribbon neighborhoods before a road vertex move"""
        usage = self.nodes.nodes.get(usage_group_id)
        if usage is None or usage.road_usage_group is None:
            self.road_vertex_prior_dirty = None
            return
        self.road_vertex_prior_dirty = dirty_aabb_for_road_endpoints(
            usage.road_usage_group,
            endpoints,
            road_parts=road_parts,
        )

    def finish_vector_edit_dirty(
        self,
        heightmap_catalog: HeightmapCatalog,
        node_ids: list[str],
    ) -> None:
        """regen vector layers with prior∪current dirty, then clear prior"""
        prior = self.vector_edit_prior_dirty
        self.vector_edit_prior_dirty = None
        regenerate_vectors_touching(
            heightmap_catalog,
            self.nodes.nodes,
            node_ids,
            prior_dirty_aabb=prior,
            defer_gpu=True,
            heightfield_gpu=self.heightfield_gpu,
        )

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

    def toggle_sea_plane(self) -> None:
        """toggle sea plane visibility"""
        self.show_sea_plane = not self.show_sea_plane

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

    def clear_selection(self, ui: UiState | None = None) -> None:
        """clear the active group selection"""
        self.nodes.clear_selection()
        if ui is not None:
            ui.clear_road_polyline_selection()
        end_gizmo_drag(self.gizmo)

    def register_mesh(self, mesh_id: MeshId, prepared: PreparedMesh) -> None:
        """insert or replace a prepared mesh in the mesh table"""
        self.mesh_table.add_or_replace(mesh_id, prepared)

    def has_mesh(self, mesh_id: MeshId) -> bool:
        """true when the mesh table already has this id"""
        return self.mesh_table.has_mesh(mesh_id)

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
        ui: UiState,
    ) -> None:
        """select a vector point, road polyline, mesh/heightfield group, or clear"""
        additive: bool = is_key_down(KeyboardKey.KEY_LEFT_CONTROL) or is_key_down(
            KeyboardKey.KEY_RIGHT_CONTROL
        )
        road_hit: RoadPolylinePickHit | None = pick_nearest_road_polyline(
            self.nodes.nodes,
            ray,
            self.camera,
            viewport_height_px,
            self.nodes.selected_ids,
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
        # road polyline beats heightfield volume; loses to closer mesh/point
        if road_hit is not None:
            beat_point: bool = (
                point_hit is None or road_hit.distance <= point_hit.distance
            )
            beat_mesh: bool = (
                mesh_hit is None or road_hit.distance <= mesh_hit.distance
            )
            if beat_point and beat_mesh:
                if not additive:
                    self.nodes.set_selection([road_hit.usage_group_id])
                elif road_hit.usage_group_id not in self.nodes.selected_ids:
                    self.nodes.set_selection([road_hit.usage_group_id])
                ui.set_road_polyline_selection(
                    road_hit.usage_group_id,
                    road_hit.polyline_id,
                    additive=additive,
                    endpoint=road_hit.endpoint,
                )
                end_gizmo_drag(self.gizmo)
                return
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
            self.clear_selection(ui)
            return
        selected_id: str = (
            root_group_id(self.nodes.nodes, hit_id) if select_as_group else hit_id
        )
        ui.clear_road_polyline_selection()
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

    def commit_road_vertex_move(
        self,
        usage_group_id: str,
        polyline_ids: Sequence[str],
        heightmap_catalog: HeightmapCatalog,
        road_parts: RoadPartsCatalog | None,
        *,
        endpoints: Sequence[RoadEndpointRef] | None = None,
    ) -> None:
        """drop stale placements, journal, rebuild pieces (ribbon if flatten)"""
        if not polyline_ids:
            self.road_vertex_prior_dirty = None
            return
        usage: Node | None = self.nodes.nodes.get(usage_group_id)
        if usage is None or usage.road_usage_group is None:
            self.road_vertex_prior_dirty = None
            return
        cleared = drop_placements(usage.road_usage_group, list(polyline_ids))
        self.nodes.add_nodes([set_usage_group_on_node(usage, cleared)])
        usage = self.nodes.nodes.get(usage_group_id)
        if usage is None or usage.road_usage_group is None:
            self.road_vertex_prior_dirty = None
            return
        if road_parts is not None:
            regenerate_road_piece_children(
                self.nodes,
                usage_group_id,
                road_parts,
                has_mesh=self.has_mesh,
                register_mesh=self.register_mesh,
                shader=self.mesh_shader(),
                heightmap_catalog=heightmap_catalog,
            )
        if usage.road_usage_group.ribbon_flatten:
            from builder.heightfield.road_sections.rasterize import (
                regenerate_road_sections_touching,
            )

            prior = self.road_vertex_prior_dirty
            self.road_vertex_prior_dirty = None
            current_dirty = None
            if endpoints is not None:
                current_dirty = dirty_aabb_for_road_endpoints(
                    usage.road_usage_group,
                    endpoints,
                    road_parts=road_parts,
                )
            dirty = union_aabb(prior, current_dirty)
            regenerate_road_sections_touching(
                heightmap_catalog,
                self.nodes.nodes,
                [usage_group_id],
                road_parts=road_parts,
                dirty_world_aabb=dirty,
                defer_gpu=True,
            )
        else:
            self.road_vertex_prior_dirty = None
        if self.nodes.journal is not None:
            self.nodes.journal.record(
                "move_road_vertex",
                usage_group_id=usage_group_id,
                polyline_ids=list(polyline_ids),
            )

    def handle_input(
        self,
        view_rect: Rectangle,
        ui_blocks_mouse: bool,
        heightmap_catalog: HeightmapCatalog,
        text_focused: bool,
        ui: UiState,
        *,
        block_left_click: bool = False,
        road_parts: RoadPartsCatalog | None = None,
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
                road_vertex: RoadVertexDrag | None = self.road_vertex_drag
                road_axis: RoadVertexAxisDrag | None = self.road_vertex_axis_drag
                if dragged:
                    snap_vector_points_to_plane(self.nodes, dragged)
                end_gizmo_drag(self.gizmo)
                self.circle_radius_drag = None
                self.road_height_drag = None
                self.road_vertex_drag = None
                self.road_vertex_axis_drag = None
                if dragged:
                    self.record_gizmo_commit(dragged)
                    regenerate_splines_touching(
                        self.nodes,
                        dragged,
                        heightmap_catalog=heightmap_catalog,
                    )
                    self.finish_vector_edit_dirty(
                        heightmap_catalog, dragged
                    )
                elif radius_shape_ids:
                    self.finish_vector_edit_dirty(
                        heightmap_catalog, radius_shape_ids
                    )
                elif road_point_ids:
                    self.finish_vector_edit_dirty(
                        heightmap_catalog, road_point_ids
                    )
                elif road_axis is not None:
                    self.commit_road_vertex_move(
                        road_axis.usage_group_id,
                        dragged_polyline_ids(road_axis),
                        heightmap_catalog,
                        road_parts,
                        endpoints=[
                            start.endpoint for start in road_axis.starts
                        ],
                    )
                elif road_vertex is not None:
                    self.commit_road_vertex_move(
                        road_vertex.usage_group_id,
                        [road_vertex.endpoint.polyline_id],
                        heightmap_catalog,
                        road_parts,
                        endpoints=[road_vertex.endpoint],
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
                self.finish_vector_edit_dirty(
                    heightmap_catalog, release_radius_ids
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
                self.finish_vector_edit_dirty(
                    heightmap_catalog, release_road_ids
                )
            return

        if self.road_vertex_axis_drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_road_vertex_axis_drag(
                    self.nodes,
                    self.road_vertex_axis_drag,
                    ray,
                    self.camera.position,
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                finished_axis: RoadVertexAxisDrag = self.road_vertex_axis_drag
                self.road_vertex_axis_drag = None
                self.commit_road_vertex_move(
                    finished_axis.usage_group_id,
                    dragged_polyline_ids(finished_axis),
                    heightmap_catalog,
                    road_parts,
                    endpoints=[
                        start.endpoint for start in finished_axis.starts
                    ],
                )
            return

        if self.road_vertex_drag is not None:
            if is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
                apply_road_vertex_drag(
                    self.nodes, self.road_vertex_drag, ray
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                finished: RoadVertexDrag = self.road_vertex_drag
                self.road_vertex_drag = None
                self.commit_road_vertex_move(
                    finished.usage_group_id,
                    [finished.endpoint.polyline_id],
                    heightmap_catalog,
                    road_parts,
                    endpoints=[finished.endpoint],
                )
            return

        pivot: Vector3 | None = selection_pivot(self.nodes, self.nodes.selected_ids)
        axes: dict[GizmoAxis, Vector3] | None = None
        size: float = 1.0
        plane_handle: bool = (
            self.gizmo.mode is GizmoMode.translate
            and selection_is_vector_workplane_nodes(
                self.nodes, self.nodes.selected_ids
            )
        )
        if (
            self.gizmo.mode is GizmoMode.rotate
            and selection_is_vector_workplane_nodes(
                self.nodes, self.nodes.selected_ids
            )
        ):
            shape_gizmo: tuple[Vector3, dict[GizmoAxis, Vector3]] | None = (
                vector_shape_rotate_pivot_and_axes(
                    self.nodes, self.nodes.selected_ids
                )
            )
            if shape_gizmo is not None:
                pivot, axes = shape_gizmo
                size = gizmo_size(self.camera, pivot)
        elif pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            axes = filter_gizmo_axes_for_selection(
                gizmo_axis_directions(self.gizmo.space, rotation),
                self.nodes,
                self.nodes.selected_ids,
                self.gizmo.mode,
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
                self.begin_vector_edit_dirty([radius_drag.shape_id])
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
                self.begin_vector_edit_dirty([height_drag.point_id])
                self.road_height_drag = height_drag
                return
            # axis gizmo then free cube drag on already-selected vertices
            if (
                ui.road_polyline_usage_group_id is not None
                and ui.road_polyline_ids
                and ui.road_endpoints
                and not (
                    is_key_down(KeyboardKey.KEY_LEFT_CONTROL)
                    or is_key_down(KeyboardKey.KEY_RIGHT_CONTROL)
                )
            ):
                axis_drag: RoadVertexAxisDrag | None = (
                    pick_road_vertex_axis_drag(
                        self.nodes.nodes,
                        ray,
                        self.camera,
                        ui.road_polyline_usage_group_id,
                        ui.road_endpoints,
                    )
                )
                if axis_drag is not None:
                    self.begin_road_vertex_edit_dirty(
                        axis_drag.usage_group_id,
                        [start.endpoint for start in axis_drag.starts],
                        road_parts,
                    )
                    self.road_vertex_axis_drag = axis_drag
                    return
                vertex_drag: RoadVertexDrag | None = pick_road_vertex_drag(
                    self.nodes.nodes,
                    ray,
                    self.camera,
                    float(view_rect.height),
                    ui.road_polyline_usage_group_id,
                    ui.road_polyline_ids,
                )
                if vertex_drag is not None:
                    selected_vertex: bool = any(
                        (
                            endpoint.polyline_id
                            == vertex_drag.endpoint.polyline_id
                            and endpoint.vertex_index
                            == vertex_drag.endpoint.vertex_index
                        )
                        for endpoint in ui.road_endpoints
                    )
                    if selected_vertex:
                        self.begin_road_vertex_edit_dirty(
                            vertex_drag.usage_group_id,
                            [vertex_drag.endpoint],
                            road_parts,
                        )
                        self.road_vertex_drag = vertex_drag
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
                self.finish_vector_edit_dirty(heightmap_catalog, dragged)
            return

        if not block_left_click and (
            is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
            and pivot is not None
            and axes is not None
        ):
            handle: GizmoAxis | None = hit_test_gizmo(
                self.gizmo,
                ray,
                pivot,
                axes,
                size,
                plane_handle=plane_handle,
            )
            if handle is not None:
                self.begin_vector_edit_dirty(
                    list(self.nodes.selected_ids)
                )
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
                ray, heightmap_catalog, float(view_rect.height), ui
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
        ui: UiState,
        *,
        sculpt_preview_center: Vector3 | None = None,
        sculpt_preview_radius: float = 0.0,
        road_parts: RoadPartsCatalog | None = None,
    ) -> None:
        """render the 3D scene into the viewport rectangle using scissor"""
        tick_deferred_gpu_flush(heightmap_catalog)
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
                show_sea=self.show_sea_plane,
            )
        node: Node
        for node in self.nodes.nodes.values():
            if node.heightfield is not None and self.line_shader is not None:
                draw_vector_overlays(
                    heightmap_catalog,
                    self.nodes.nodes,
                    node,
                    self.camera,
                    float(view_rect.height),
                    self.line_shader,
                    self.line_batch,
                    self.nodes.selected_ids,
                    heightfield_gpu=self.heightfield_gpu,
                )
                live_road_ids: list[str] = []
                if self.road_vertex_axis_drag is not None:
                    live_road_ids = dragged_polyline_ids(
                        self.road_vertex_axis_drag
                    )
                elif self.road_vertex_drag is not None:
                    live_road_ids = [
                        self.road_vertex_drag.endpoint.polyline_id
                    ]
                draw_road_sections_overlays(
                    heightmap_catalog,
                    self.nodes.nodes,
                    node,
                    self.line_shader,
                    self.line_batch,
                    self.road_line_mesh_cache,
                    self.nodes.selected_ids,
                    ui.road_polyline_ids,
                    ui.road_endpoints,
                    camera=self.camera,
                    viewport_height_px=float(view_rect.height),
                    road_parts=road_parts,
                    live_polyline_ids=live_road_ids,
                    heightfield_gpu=self.heightfield_gpu,
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
        draw_axes: dict[GizmoAxis, Vector3] | None = None
        draw_plane_handle: bool = (
            self.gizmo.mode is GizmoMode.translate
            and selection_is_vector_workplane_nodes(
                self.nodes, self.nodes.selected_ids
            )
        )
        if (
            self.gizmo.mode is GizmoMode.rotate
            and selection_is_vector_workplane_nodes(
                self.nodes, self.nodes.selected_ids
            )
        ):
            shape_draw: tuple[Vector3, dict[GizmoAxis, Vector3]] | None = (
                vector_shape_rotate_pivot_and_axes(
                    self.nodes, self.nodes.selected_ids
                )
            )
            if shape_draw is not None:
                pivot, draw_axes = shape_draw
        elif pivot is not None:
            rotation: Quaternion = selection_rotation(
                self.nodes, self.nodes.selected_ids
            )
            draw_axes = filter_gizmo_axes_for_selection(
                gizmo_axis_directions(self.gizmo.space, rotation),
                self.nodes,
                self.nodes.selected_ids,
                self.gizmo.mode,
            )
        if pivot is not None and draw_axes is not None:
            draw_gizmo(
                self.gizmo,
                pivot,
                draw_axes,
                gizmo_size(self.camera, pivot),
                plane_handle=draw_plane_handle,
            )
        if (
            ui.road_polyline_usage_group_id is not None
            and ui.road_endpoints
        ):
            vertex_context = selection_pivot_world(
                self.nodes.nodes,
                ui.road_polyline_usage_group_id,
                ui.road_endpoints,
            )
            if vertex_context is not None:
                vertex_pivot: Vector3
                hf_world_gizmo: Matrix
                vertex_pivot, hf_world_gizmo, _, _ = vertex_context
                draw_road_vertex_move_gizmo(
                    self.camera,
                    vertex_pivot,
                    hf_local_xz_axes(hf_world_gizmo),
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
        clear_line_mesh_cache(self.road_line_mesh_cache)
        if self.line_shader is not None:
            unload_line_shader(self.line_shader)
            self.line_shader = None
        self.lighting.unload()
