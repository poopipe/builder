"""application session and main window loop"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyray import (
    ConfigFlags,
    FilePathList,
    Font,
    KeyboardKey,
    MouseButton,
    Rectangle,
    Vector2,
    Vector3,
    begin_drawing,
    clear_background,
    close_window,
    end_drawing,
    ffi,
    get_fps,
    get_frame_time,
    get_mouse_position,
    get_screen_height,
    get_screen_width,
    init_window,
    is_file_dropped,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    is_window_ready,
    load_dropped_files,
    set_config_flags,
    set_exit_key,
    set_target_fps,
    unload_dropped_files,
    window_should_close,
)

from builder.application.application_state import ApplicationState
from builder.commands.commands_types import CommandEntry, CommandItem
from builder.commands.file import (
    apply_import_heightmap_path,
    apply_import_mesh_path,
    apply_open_scene_path,
    apply_save_scene_path,
    cmd_import_mesh,
    import_mesh_from_path,
    remember_browser_directory,
)
from builder.commands.menus import menu_commands, panel_commands, all_commands
from builder.commands.registry import CommandRegistry, bind_entry, register_commands
from builder.commands.scene import (
    cmd_group,
    cmd_parent,
    cmd_select_mesh,
    cmd_unparent,
)
from builder.generators.regenerate import selected_parametric_group
from builder.heightfield.regenerate import selected_or_ancestor_heightfield_group
from builder.scene.scene_types import MeshId, Node
from builder.session_journal import log_exception
from builder.ui.file_browser import (
    BrowserRow,
    FileBrowserCancel,
    FileBrowserFrameResult,
    FileBrowserPurpose,
    draw_file_browser,
    update_file_browser,
)
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.inspector import (
    ParamRowRects,
    draw_inspector,
    sync_inspector_focus,
    update_inspector,
)
from builder.ui.vector_layer_inspector import (
    draw_vector_layer_inspector,
    selected_vector_layer_group,
    update_vector_layer_inspector,
)
from builder.ui.vector_shape_inspector import (
    VectorShapeFloatRow,
    draw_vector_shape_inspector,
    selected_vector_shape_group,
    update_vector_shape_inspector,
)
from builder.ui.modifiers_panel import (
    ModifierPanelRow,
    draw_modifiers_panel,
    sync_modifiers_focus,
    update_modifiers_panel,
)
from builder.heightfield.sculpt_stroke import (
    FieldUvHit,
    SculptStrokeState,
    SculptToolTarget,
    apply_sculpt_stamp,
    begin_sculpt_stroke,
    pick_field_uv,
    selected_sculpt_tool_target,
)
from builder.ui.height_layers_panel import (
    LayerFloatRow,
    LayerListRow,
    draw_height_layers_panel,
    update_height_layers_panel,
)
from builder.ui.mesh_panel import MeshRow, draw_mesh_panel, update_mesh_panel
from builder.ui.sculpt_panel import (
    SculptBrushRow,
    SculptFloatRow,
    draw_sculpt_panel,
    update_sculpt_panel,
)
from builder.ui.outliner import (
    OutlinerRow,
    draw_outliner,
    sync_outliner_focus,
    update_outliner,
)
from builder.ui.snap_bar import (
    SnapBarRects,
    clear_snap_focus,
    draw_snap_bar,
    layout_snap_bar,
    update_snap_bar,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_button_min_width,
    ui_color_bg,
    ui_font_size,
    ui_height_layers_panel_width,
    ui_inspector_panel_width,
    ui_meshes_panel_width,
    ui_modifiers_panel_width,
    ui_outliner_panel_width,
    ui_pad,
    ui_sculpt_panel_width,
    ui_side_panel_width,
)
from builder.ui.transform_bar import (
    TransformBarRects,
    clear_transform_focus,
    draw_transform_bar,
    layout_transform_bar,
    update_transform_bar,
)
from builder.ui.ui_state import UiState
from builder.ui.move_bar import (
    MoveBarRects,
    clear_move_focus,
    draw_move_bar,
    layout_move_bar,
    update_move_bar,
)
from builder.ui.view_distance_bar import (
    ViewDistanceBarRects,
    clear_view_distance_focus,
    draw_view_distance_bar,
    layout_view_distance_bar,
    update_view_distance_bar,
)
from builder.ui.zoom_bar import (
    ZoomBarRects,
    clear_zoom_focus,
    draw_zoom_bar,
    layout_zoom_bar,
    update_zoom_bar,
)
from builder.ui.widgets import (
    Button,
    LayoutRects,
    compute_layout,
    draw_button_stack,
    draw_menu_bar,
    draw_status_bar,
    is_point_in_rect,
    layout_buttons_horizontal,
    layout_buttons_vertical,
    update_buttons,
)
from builder.view.viewport import Viewport
from builder.view.viewport_ray import viewport_world_ray


def dropped_path_string(raw: Any) -> str:
    """decode a FilePathList.paths entry (str, bytes, or cffi char*)"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="surrogateescape")
    # cffi char* — str(cdata) is a pointer repr, not the path text
    text: bytes | str = ffi.string(raw)
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="surrogateescape")
    return text


def apply_text_exit_key(ui: UiState) -> None:
    """disable window-close-on-escape while typing or a file dialog is open"""
    if ui.focus is not None or ui.file_browser is not None:
        set_exit_key(0)
    else:
        set_exit_key(KeyboardKey.KEY_ESCAPE)


class Application:
    """owns session state and runs the main loop

    Structurally satisfies ``CommandContext`` (application, scene, ui)
    Construct only after the raylib window exists so ``scene`` and ``font``
    can be created up front
    """

    def __init__(self, scene: Viewport, font: Font) -> None:
        self.application: ApplicationState = ApplicationState()
        self.scene: Viewport = scene
        self.scene.nodes.journal = self.application.journal
        self.ui: UiState = UiState(status=self.application.importer.status_message())
        self.font: Font = font
        self.commands: CommandRegistry = register_commands(all_commands())
        self.sculpt_stroke: SculptStrokeState | None = None
        self.sculpt_hover: FieldUvHit | None = None

    def command_button_items(
        self, entries: tuple[CommandItem, ...]
    ) -> list[tuple[str, Callable[[], None]]]:
        items: list[tuple[str, Callable[[], None]]] = []
        entry: CommandItem
        for entry in entries:
            items.append((entry.command.label, bind_entry(self, entry)))
        return items

    def handle_dropped_files(self) -> None:
        """import the first dropped .fbx, or report why nothing was imported"""
        if not is_file_dropped():
            return
        dropped: FilePathList = load_dropped_files()
        try:
            count: int = int(dropped.count)
            if count <= 0:
                return
            seen: list[str] = []
            index: int
            for index in range(count):
                path: str = dropped_path_string(dropped.paths[index])
                seen.append(path)
                if not self.application.importer.has_valid_suffix(path):
                    continue
                try:
                    self.application.journal.record("import_mesh", path=path)
                    import_mesh_from_path(self, path)
                except (OSError, RuntimeError, ValueError) as exc:
                    log_exception("dropped mesh import failed", exc)
                    self.ui.status = f"Import failed: {exc}"
                return
            sample: str = Path(seen[0]).name if seen else "?"
            self.ui.status = (
                f"Dropped {count} file(s); none were .fbx " f"(got '{sample}')"
            )
        finally:
            unload_dropped_files(dropped)

    def frame(self) -> None:
        """define the UI layout"""
        self.handle_dropped_files()
        group: Node | None = selected_parametric_group(self.scene.nodes)
        if group is None:
            group = selected_or_ancestor_heightfield_group(self.scene.nodes)
        vector_shape: Node | None = selected_vector_shape_group(self.scene.nodes)
        vector_layer: Node | None = None
        if vector_shape is None:
            vector_layer = selected_vector_layer_group(self.scene.nodes)
        sync_inspector_focus(self.ui, group)
        sync_modifiers_focus(self.ui, group)
        apply_text_exit_key(self.ui)

        panel_width: int = ui_side_panel_width if self.ui.side_panel_open else 0
        outliner_width: int = ui_outliner_panel_width if self.ui.outliner_open else 0
        meshes_width: int = ui_meshes_panel_width if self.ui.meshes_panel_open else 0
        height_layers_width: int = (
            ui_height_layers_panel_width if self.ui.height_layers_panel_open else 0
        )
        sculpt_width: int = (
            ui_sculpt_panel_width if self.ui.sculpt_panel_open else 0
        )
        inspector_width: int = (
            ui_inspector_panel_width
            if group is not None
            or vector_shape is not None
            or vector_layer is not None
            else 0
        )
        modifiers_width: int = (
            ui_modifiers_panel_width
            if group is not None and group.generator is not None
            else 0
        )
        layout: LayoutRects = compute_layout(
            get_screen_width(),
            get_screen_height(),
            panel_width=panel_width,
            outliner_width=outliner_width,
            meshes_width=meshes_width,
            height_layers_width=height_layers_width,
            sculpt_width=sculpt_width,
            inspector_width=inspector_width,
            modifiers_width=modifiers_width,
        )

        browser_open: bool = self.ui.file_browser is not None
        menu_buttons: list[Button] = layout_buttons_horizontal(
            self.font,
            layout.menu,
            self.command_button_items(menu_commands),
            pad=float(ui_pad),
            gap=float(ui_button_gap),
            button_height=float(ui_button_height),
            min_width=float(ui_button_min_width),
            font_size=float(ui_font_size),
        )
        if not browser_open:
            update_buttons(menu_buttons, layout.menu)

        snap_bar: SnapBarRects | None = None
        snap_buttons: list[Button] = []
        transform_bar: TransformBarRects | None = None
        zoom_bar: ZoomBarRects | None = None
        zoom_buttons: list[Button] = []
        move_bar: MoveBarRects | None = None
        move_buttons: list[Button] = []
        view_bar: ViewDistanceBarRects | None = None
        view_buttons: list[Button] = []
        if not browser_open:
            menu_right: float = float(ui_pad)
            if menu_buttons:
                last_menu: Button = menu_buttons[-1]
                menu_right = (
                    last_menu.rect.x + last_menu.rect.width + float(ui_button_gap)
                )
            snap_bar = layout_snap_bar(self.font, layout.menu, menu_right)
            if snap_bar is None:
                clear_snap_focus(self.ui)
            else:
                snap_buttons = update_snap_bar(self.scene.gizmo, self.ui, snap_bar)
                menu_right = (
                    snap_bar.area.x + snap_bar.area.width + float(ui_button_gap)
                )
            # zoom/move/view sit in the left cluster after snap; transform stays right-aligned
            zoom_bar = layout_zoom_bar(self.font, layout.menu, menu_right)
            if zoom_bar is None:
                clear_zoom_focus(self.ui)
            else:
                zoom_buttons = update_zoom_bar(self.scene.orbit, self.ui, zoom_bar)
                menu_right = (
                    zoom_bar.area.x + zoom_bar.area.width + float(ui_button_gap)
                )
            move_bar = layout_move_bar(self.font, layout.menu, menu_right)
            if move_bar is None:
                clear_move_focus(self.ui)
            else:
                move_buttons = update_move_bar(self.scene.orbit, self.ui, move_bar)
                menu_right = (
                    move_bar.area.x + move_bar.area.width + float(ui_button_gap)
                )
            view_bar = layout_view_distance_bar(self.font, layout.menu, menu_right)
            if view_bar is None:
                clear_view_distance_focus(self.ui)
            else:
                view_buttons = update_view_distance_bar(
                    self.scene.orbit, self.ui, view_bar
                )
                menu_right = (
                    view_bar.area.x + view_bar.area.width + float(ui_button_gap)
                )
            transform_bar = layout_transform_bar(
                self.font,
                layout.menu,
                self.scene.nodes,
                self.scene.gizmo.mode,
                menu_right,
            )
            if transform_bar is None:
                clear_transform_focus(self.ui)
            else:
                update_transform_bar(
                    self.scene.nodes,
                    self.ui,
                    self.scene.gizmo.mode,
                    self.scene.gizmo.space,
                    transform_bar,
                )

        # buttons run commands
        # commands are listed in consts (eg. panel_commands)
        # executable code for command lives in a file per context (eg. view.py)
        # CommandContext gives the command access to the app state
        # (application, scene, ui)

        panel_buttons: list[Button] = []
        if self.ui.side_panel_open:
            panel_buttons = layout_buttons_vertical(
                layout.panel,
                self.command_button_items(panel_commands),
                pad=float(ui_pad),
                gap=float(ui_button_gap),
                button_height=float(ui_button_height),
            )
            if not browser_open:
                update_buttons(panel_buttons, layout.panel)

        outliner_rows: list[OutlinerRow] = []
        outliner_buttons: list[Button] = []
        if self.ui.outliner_open and not browser_open:
            sync_outliner_focus(self.scene.nodes, self.ui)
            outliner_rows, outliner_buttons = update_outliner(
                self.scene.nodes,
                self.ui,
                layout.outliner,
                self.application.mesh_catalog,
                bind_entry(self, CommandEntry(cmd_group, None)),
                bind_entry(self, CommandEntry(cmd_parent, None)),
                bind_entry(self, CommandEntry(cmd_unparent, None)),
            )

        mesh_rows: list[MeshRow] = []
        mesh_buttons: list[Button] = []
        if self.ui.meshes_panel_open and not browser_open:

            def select_active_mesh(mesh_id: MeshId) -> None:
                bind_entry(self, CommandEntry(cmd_select_mesh, mesh_id))()

            mesh_rows, mesh_buttons = update_mesh_panel(
                self.application.mesh_catalog,
                self.application.active_mesh_id,
                self.ui,
                layout.meshes,
                select_active_mesh,
                bind_entry(self, CommandEntry(cmd_import_mesh, None)),
            )

        height_layer_rows: list[LayerListRow] = []
        height_layer_float_rows: list[LayerFloatRow] = []
        height_layer_buttons: list[Button] = []
        height_layers_blend_label: Rectangle | None = None
        height_layers_empty: str | None = None
        heightfield_group: Node | None = selected_or_ancestor_heightfield_group(
            self.scene.nodes
        )
        if self.ui.height_layers_panel_open and not browser_open:
            (
                height_layer_rows,
                height_layer_float_rows,
                height_layer_buttons,
                height_layers_blend_label,
                height_layers_empty,
            ) = update_height_layers_panel(
                self.scene,
                self.application.heightmap_catalog,
                self.ui,
                layout.height_layers,
                heightfield_group,
                journal=self.application.journal,
            )
            heightfield_group = selected_or_ancestor_heightfield_group(
                self.scene.nodes
            )

        sculpt_brush_rows: list[SculptBrushRow] = []
        sculpt_float_rows: list[SculptFloatRow] = []
        sculpt_buttons: list[Button] = []
        if self.ui.sculpt_panel_open and not browser_open:
            sculpt_brush_rows, sculpt_float_rows, sculpt_buttons = update_sculpt_panel(
                self.ui, layout.sculpt
            )

        inspector_buttons: list[Button] = []
        inspector_rows: list[ParamRowRects] = []
        vector_inspector_buttons: list[Button] = []
        vector_float_rows: list[VectorShapeFloatRow] = []
        vector_layer_buttons: list[Button] = []
        modifiers_buttons: list[Button] = []
        modifiers_rows: list[ModifierPanelRow] = []
        if vector_shape is not None and not browser_open:
            vector_inspector_buttons, vector_float_rows = (
                update_vector_shape_inspector(
                    self.scene.nodes,
                    self.application.heightmap_catalog,
                    self.ui,
                    layout.inspector,
                    vector_shape,
                )
            )
            refreshed_shape: Node | None = self.scene.nodes.nodes.get(
                vector_shape.id
            )
            vector_shape = (
                refreshed_shape
                if refreshed_shape is not None
                and refreshed_shape.vector_shape is not None
                else None
            )
        elif vector_layer is not None and not browser_open:
            vector_layer_buttons = update_vector_layer_inspector(
                self.scene.nodes,
                self.application.heightmap_catalog,
                self.ui,
                layout.inspector,
                vector_layer,
            )
            # add-shape may change selection to the new shape
            vector_shape = selected_vector_shape_group(self.scene.nodes)
            if vector_shape is not None:
                vector_layer = None
                vector_inspector_buttons, vector_float_rows = (
                    update_vector_shape_inspector(
                        self.scene.nodes,
                        self.application.heightmap_catalog,
                        self.ui,
                        layout.inspector,
                        vector_shape,
                    )
                )
            else:
                refreshed_layer: Node | None = self.scene.nodes.nodes.get(
                    vector_layer.id
                )
                vector_layer = (
                    refreshed_layer
                    if refreshed_layer is not None
                    and refreshed_layer.vector_layer is not None
                    else None
                )
        elif group is not None and not browser_open:
            group_id: str = group.id
            inspector_buttons, inspector_rows = update_inspector(
                self.scene.nodes,
                self.ui,
                layout.inspector,
                group,
                self.application.mesh_catalog,
                self.application.active_mesh_id,
                scene_context=self.scene,
                heightmap_catalog=self.application.heightmap_catalog,
                journal=self.application.journal,
            )
            # steppers/bake replace the group node; re-read the same id so the
            # drawn group stays consistent with rows even if selection changed
            refreshed: Node | None = self.scene.nodes.nodes.get(group_id)
            if refreshed is None or (
                refreshed.generator is None and refreshed.heightfield is None
            ):
                group = None
            else:
                group = refreshed
            if group is not None and group.generator is not None:
                modifiers_buttons, modifiers_rows = update_modifiers_panel(
                    self.scene.nodes,
                    self.ui,
                    layout.modifiers,
                    group,
                    journal=self.application.journal,
                )
                refreshed = self.scene.nodes.nodes.get(group_id)
                if refreshed is None or (
                    refreshed.generator is None and refreshed.heightfield is None
                ):
                    group = None
                else:
                    group = refreshed

        browser_rows: list[BrowserRow] = []
        browser_buttons: list[Button] = []
        browser_window: Rectangle | None = None
        if self.ui.file_browser is not None:
            browser_result: FileBrowserFrameResult
            browser_result, browser_rows, browser_buttons, browser_window = (
                update_file_browser(
                    self.ui.file_browser,
                    get_screen_width(),
                    get_screen_height(),
                )
            )
            match browser_result:
                case FileBrowserCancel.cancelled:
                    remember_browser_directory(self, self.ui.file_browser)
                    self.ui.file_browser = None
                    self.ui.heightfield_layer_target = None
                    self.ui.heightfield_layer_replace_index = None
                    self.ui.status = "Cancelled"
                case Path() as chosen:
                    purpose: FileBrowserPurpose = self.ui.file_browser.purpose
                    remember_browser_directory(self, self.ui.file_browser)
                    self.ui.file_browser = None
                    match purpose:
                        case FileBrowserPurpose.open_scene:
                            apply_open_scene_path(self, chosen)
                        case FileBrowserPurpose.save_scene:
                            apply_save_scene_path(self, chosen)
                        case FileBrowserPurpose.import_mesh:
                            apply_import_mesh_path(self, chosen)
                        case FileBrowserPurpose.import_heightmap:
                            apply_import_heightmap_path(self, chosen)
                case None:
                    pass

        mouse: Vector2 = get_mouse_position()
        ui_over: bool = browser_open or (
            is_point_in_rect(mouse.x, mouse.y, layout.menu)
            or (
                self.ui.side_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.panel)
            )
            or (
                self.ui.outliner_open
                and is_point_in_rect(mouse.x, mouse.y, layout.outliner)
            )
            or (
                self.ui.meshes_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.meshes)
            )
            or (
                self.ui.height_layers_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.height_layers)
            )
            or (
                self.ui.sculpt_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.sculpt)
            )
            or (
                group is not None
                and is_point_in_rect(mouse.x, mouse.y, layout.inspector)
            )
            or (
                group is not None
                and group.generator is not None
                and is_point_in_rect(mouse.x, mouse.y, layout.modifiers)
            )
            or is_point_in_rect(mouse.x, mouse.y, layout.status)
        )
        sculpting: bool = False
        sculpt_target: SculptToolTarget | None = None
        self.sculpt_hover = None
        if (
            self.ui.sculpt_panel_open
            and heightfield_group is not None
            and heightfield_group.heightfield is not None
        ):
            sculpt_target = selected_sculpt_tool_target(
                heightfield_group, self.ui.height_layers_selected_index
            )
            sculpting = sculpt_target is not None
        if sculpting and not ui_over and sculpt_target is not None:
            assert heightfield_group is not None
            assert heightfield_group.heightfield is not None
            ray = viewport_world_ray(self.scene.camera, mouse)
            hit: FieldUvHit | None = pick_field_uv(
                self.scene.nodes,
                self.application.heightmap_catalog,
                self.scene.draw_cache,
                ray,
                heightfield_group.id,
            )
            self.sculpt_hover = hit
            if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
                if hit is not None:
                    self.sculpt_stroke = begin_sculpt_stroke(
                        self.application.heightmap_catalog,
                        heightfield_group.heightfield,
                        sculpt_target,
                        heightfield_group.id,
                        hit.u,
                        hit.v,
                    )
                    apply_sculpt_stamp(
                        self.application.heightmap_catalog,
                        heightfield_group.heightfield,
                        self.sculpt_stroke,
                        self.ui.sculpt_brush_kind,
                        hit.u,
                        hit.v,
                        self.ui.sculpt_radius,
                        self.ui.sculpt_intensity,
                    )
                    self.ui.status = (
                        f"Sculpt {self.ui.sculpt_brush_kind} "
                        f"r={self.ui.sculpt_radius:g}"
                    )
                else:
                    self.ui.status = "Sculpt: miss heightfield"
            elif (
                is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
                and self.sculpt_stroke is not None
                and hit is not None
            ):
                apply_sculpt_stamp(
                    self.application.heightmap_catalog,
                    heightfield_group.heightfield,
                    self.sculpt_stroke,
                    self.ui.sculpt_brush_kind,
                    hit.u,
                    hit.v,
                    self.ui.sculpt_radius,
                    self.ui.sculpt_intensity,
                )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                self.sculpt_stroke = None
        elif self.sculpt_stroke is not None:
            self.sculpt_stroke = None
        self.scene.handle_input(
            layout.viewport,
            ui_over,
            self.application.heightmap_catalog,
            text_focused=self.ui.focus is not None,
            block_left_click=sculpting and not ui_over,
        )

        begin_drawing()
        clear_background(ui_color_bg)
        preview_center: Vector3 | None = (
            None if self.sculpt_hover is None else self.sculpt_hover.world_point
        )
        preview_radius: float = (
            self.ui.sculpt_radius if self.sculpt_hover is not None else 0.0
        )
        self.scene.draw(
            layout.viewport,
            self.application.heightmap_catalog,
            sculpt_preview_center=preview_center,
            sculpt_preview_radius=preview_radius,
        )
        draw_menu_bar(self.font, layout.menu, menu_buttons)
        if snap_bar is not None:
            draw_snap_bar(self.font, self.scene.gizmo, self.ui, snap_bar, snap_buttons)
        if transform_bar is not None:
            draw_transform_bar(
                self.font,
                self.scene.nodes,
                self.ui,
                self.scene.gizmo.mode,
                self.scene.gizmo.space,
                transform_bar,
            )
        if zoom_bar is not None:
            draw_zoom_bar(
                self.font, self.scene.orbit, self.ui, zoom_bar, zoom_buttons
            )
        if move_bar is not None:
            draw_move_bar(
                self.font, self.scene.orbit, self.ui, move_bar, move_buttons
            )
        if view_bar is not None:
            draw_view_distance_bar(
                self.font, self.scene.orbit, self.ui, view_bar, view_buttons
            )
        if self.ui.side_panel_open:
            draw_button_stack(self.font, layout.panel, panel_buttons)
        if self.ui.outliner_open:
            draw_outliner(
                self.font, layout.outliner, self.ui, outliner_rows, outliner_buttons
            )
        if self.ui.meshes_panel_open:
            draw_mesh_panel(self.font, layout.meshes, mesh_rows, mesh_buttons)
        if self.ui.height_layers_panel_open:
            draw_height_layers_panel(
                self.font,
                layout.height_layers,
                self.ui,
                heightfield_group,
                height_layer_rows,
                height_layer_float_rows,
                height_layer_buttons,
                height_layers_blend_label,
                height_layers_empty,
            )
        if self.ui.sculpt_panel_open:
            draw_sculpt_panel(
                self.font,
                layout.sculpt,
                self.ui,
                sculpt_brush_rows,
                sculpt_float_rows,
                sculpt_buttons,
            )
        if vector_shape is not None:
            draw_vector_shape_inspector(
                self.font,
                layout.inspector,
                self.ui,
                vector_shape,
                vector_inspector_buttons,
                vector_float_rows,
            )
        elif vector_layer is not None:
            draw_vector_layer_inspector(
                self.font,
                layout.inspector,
                vector_layer,
                vector_layer_buttons,
            )
        elif group is not None:
            draw_inspector(
                self.font,
                layout.inspector,
                self.ui,
                group,
                inspector_buttons,
                inspector_rows,
            )
            if group.generator is not None:
                draw_modifiers_panel(
                    self.font,
                    layout.modifiers,
                    self.ui,
                    group,
                    modifiers_buttons,
                    modifiers_rows,
                )
        draw_status_bar(
            self.font,
            layout.status,
            self.ui.status,
            fps=get_fps(),
            frame_ms=get_frame_time() * 1000.0,
        )
        if self.ui.file_browser is not None and browser_window is not None:
            draw_file_browser(
                self.font,
                self.ui.file_browser,
                browser_rows,
                browser_buttons,
                browser_window,
                get_screen_width(),
                get_screen_height(),
            )
        end_drawing()

    def run(self) -> None:
        while not window_should_close() and not self.application.should_close:
            try:
                self.frame()
            except Exception as exc:
                log_exception("frame failed", exc)
                raise

    def shutdown(self) -> None:
        """release GPU resources owned by this session"""
        self.scene.unload()
        unload_app_font(self.font)


def open_window() -> None:
    """create the raylib window (required before GPU resources)"""
    set_config_flags(ConfigFlags.FLAG_WINDOW_RESIZABLE | ConfigFlags.FLAG_MSAA_4X_HINT)
    init_window(1280, 720, "Builder")
    if not is_window_ready():
        raise RuntimeError("failed to create application window")
    set_target_fps(60)
    set_exit_key(KeyboardKey.KEY_ESCAPE)


def run_application() -> None:
    """open the window, build the application, run until quit, then tear down"""
    open_window()
    font: Font = load_app_font(ui_font_size)

    scene: Viewport
    try:
        scene = Viewport()
    except RuntimeError as exc:
        log_exception("viewport init failed", exc)
        unload_app_font(font)
        close_window()
        raise

    app: Application = Application(scene=scene, font=font)
    app.application.journal.begin_session()
    try:
        app.run()
    except Exception as exc:
        log_exception("application crashed", exc)
        raise
    finally:
        app.application.journal.close()
        app.shutdown()
        close_window()
