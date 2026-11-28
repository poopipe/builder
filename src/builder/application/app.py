"""application session and main window loop"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
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
    apply_export_heightmap_path,
    apply_export_tb_roads_path,
    apply_import_heightfield_albedo_path,
    apply_import_heightmap_path,
    apply_import_mesh_path,
    apply_import_road_shp_path,
    apply_import_tb_objects_path,
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
from builder.ui.color_palette_popup import (
    ColorPaletteCancel,
    ColorPaletteClear,
    ColorPaletteFrameResult,
    ColorPaletteSwatch,
    draw_color_palette_popup,
    update_color_palette_popup,
)
from builder.ui.list_picker import (
    ListPickerCancel,
    ListPickerFrameResult,
    ListPickerPurpose,
    ListPickerRow,
    draw_list_picker,
    list_picker_new_item,
    update_list_picker,
)
from builder.ui.outliner import set_node_color_index
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.inspector import (
    ParamRowRects,
    draw_inspector,
    sync_inspector_focus,
    update_inspector,
)
from builder.heightfield.road_sections.pick import (
    clear_road_polyline_selection_if_stale,
)
from builder.ui.road_sections_inspector import (
    RoadSectionsInspectorLayout,
    apply_move_polylines_to_usage,
    apply_road_kit_choice,
    begin_move_usage_name,
    draw_road_sections_inspector,
    selected_road_sections_layer,
    update_road_sections_inspector,
)
from builder.ui.vector_layer_inspector import (
    draw_vector_layer_inspector,
    selected_vector_layer_group,
    update_vector_layer_inspector,
)
from builder.ui.vector_shape_inspector import (
    VectorShapeFloatRow,
    apply_vector_brush_choice,
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
from builder.heightfield.heightfield_types import Heightfield
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
from builder.ui.dock import (
    DockPanel,
    RightDockContent,
    available_left_tabs,
    available_right_tabs,
    clamp_dock_width,
    draw_tab_bar,
    resolve_active_tab,
    update_dock_resize,
    update_tab_bar,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_button_min_width,
    ui_color_bg,
    ui_font_size,
    ui_pad,
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
    """disable window-close-on-escape while typing or a modal dialog is open"""
    if (
        ui.focus is not None
        or ui.file_browser is not None
        or ui.list_picker is not None
        or ui.color_palette is not None
    ):
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
        road_sections_layer: Node | None = None
        if vector_shape is None:
            vector_layer = selected_vector_layer_group(self.scene.nodes)
        if vector_shape is None and vector_layer is None:
            road_sections_layer = selected_road_sections_layer(self.scene.nodes)
        sync_inspector_focus(self.ui, group)
        sync_modifiers_focus(self.ui, group)
        apply_text_exit_key(self.ui)
        if clear_road_polyline_selection_if_stale(
            self.scene.nodes.nodes,
            self.ui.road_polyline_usage_group_id,
            self.ui.road_polyline_ids,
            self.scene.nodes.selected_ids,
        ):
            self.ui.clear_road_polyline_selection()

        show_inspector: bool = (
            group is not None
            or vector_shape is not None
            or vector_layer is not None
            or road_sections_layer is not None
        )
        show_modifiers: bool = group is not None and group.generator is not None
        right_content: RightDockContent = RightDockContent(
            show_inspector=show_inspector,
            show_modifiers=show_modifiers,
        )
        left_tabs: list[DockPanel] = available_left_tabs(
            side_panel_open=self.ui.side_panel_open,
            outliner_open=self.ui.outliner_open,
        )
        right_tabs: list[DockPanel] = available_right_tabs(
            meshes_panel_open=self.ui.meshes_panel_open,
            height_layers_panel_open=self.ui.height_layers_panel_open,
            sculpt_panel_open=self.ui.sculpt_panel_open,
            content=right_content,
        )
        prefer_right: DockPanel | None = None
        if show_inspector:
            prefer_right = DockPanel.inspector
        elif show_modifiers:
            prefer_right = DockPanel.modifiers
        resolved_left: DockPanel | None = resolve_active_tab(
            self.ui.left_dock_tab, left_tabs
        )
        resolved_right: DockPanel | None = resolve_active_tab(
            self.ui.right_dock_tab, right_tabs, prefer_right
        )
        if resolved_left is not None:
            self.ui.left_dock_tab = resolved_left
        if resolved_right is not None:
            self.ui.right_dock_tab = resolved_right
        self.ui.left_dock_width = clamp_dock_width(self.ui.left_dock_width)
        self.ui.right_dock_width = clamp_dock_width(self.ui.right_dock_width)
        left_width: float = self.ui.left_dock_width if left_tabs else 0.0
        right_width: float = self.ui.right_dock_width if right_tabs else 0.0
        layout: LayoutRects = compute_layout(
            get_screen_width(),
            get_screen_height(),
            left_width=left_width,
            right_width=right_width,
        )

        modal_open: bool = (
            self.ui.file_browser is not None
            or self.ui.list_picker is not None
            or self.ui.color_palette is not None
        )
        if not modal_open:
            resized_left: float
            resized_right: float
            resized_left, resized_right = update_dock_resize(
                self.ui.dock_resize,
                left_dock=layout.left_dock,
                right_dock=layout.right_dock,
                left_width=self.ui.left_dock_width,
                right_width=self.ui.right_dock_width,
            )
            if left_tabs:
                self.ui.left_dock_width = resized_left
            if right_tabs:
                self.ui.right_dock_width = resized_right
            width_changed: bool = (
                abs(resized_left - left_width) > 0.01
                or abs(resized_right - right_width) > 0.01
            )
            if width_changed:
                left_width = self.ui.left_dock_width if left_tabs else 0.0
                right_width = self.ui.right_dock_width if right_tabs else 0.0
                layout = compute_layout(
                    get_screen_width(),
                    get_screen_height(),
                    left_width=left_width,
                    right_width=right_width,
                )
            selected_left: DockPanel | None = update_tab_bar(
                layout.left_dock, left_tabs, resolved_left
            )
            selected_right: DockPanel | None = update_tab_bar(
                layout.right_dock, right_tabs, resolved_right
            )
            if selected_left is not None:
                self.ui.left_dock_tab = selected_left
                resolved_left = selected_left
            if selected_right is not None:
                self.ui.right_dock_tab = selected_right
                resolved_right = selected_right
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
        if not modal_open:
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
        if not modal_open:
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
                    heightmap_catalog=self.application.heightmap_catalog,
                )

        # buttons run commands
        # commands are listed in consts (eg. panel_commands)
        # executable code for command lives in a file per context (eg. view.py)
        # CommandContext gives the command access to the app state
        # (application, scene, ui)

        left_content: Rectangle = layout.left_content
        right_content_rect: Rectangle = layout.right_content
        active_left: DockPanel | None = resolved_left
        active_right: DockPanel | None = resolved_right

        panel_buttons: list[Button] = []
        if active_left is DockPanel.tools:
            panel_buttons = layout_buttons_vertical(
                left_content,
                self.command_button_items(panel_commands),
                pad=float(ui_pad),
                gap=float(ui_button_gap),
                button_height=float(ui_button_height),
            )
            if not modal_open:
                update_buttons(panel_buttons, left_content)

        outliner_rows: list[OutlinerRow] = []
        outliner_buttons: list[Button] = []
        if active_left is DockPanel.outliner and not modal_open:
            sync_outliner_focus(self.scene.nodes, self.ui)
            outliner_rows, outliner_buttons = update_outliner(
                self.scene.nodes,
                self.ui,
                left_content,
                self.application.mesh_catalog,
                bind_entry(self, CommandEntry(cmd_group, None)),
                bind_entry(self, CommandEntry(cmd_parent, None)),
                bind_entry(self, CommandEntry(cmd_unparent, None)),
            )

        mesh_rows: list[MeshRow] = []
        mesh_buttons: list[Button] = []
        if active_right is DockPanel.meshes and not modal_open:

            def select_active_mesh(mesh_id: MeshId) -> None:
                bind_entry(self, CommandEntry(cmd_select_mesh, mesh_id))()

            mesh_rows, mesh_buttons = update_mesh_panel(
                self.application.mesh_catalog,
                self.application.active_mesh_id,
                self.ui,
                right_content_rect,
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
        if active_right is DockPanel.height_layers and not modal_open:
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
                right_content_rect,
                heightfield_group,
                journal=self.application.journal,
            )
            heightfield_group = selected_or_ancestor_heightfield_group(
                self.scene.nodes
            )

        sculpt_brush_rows: list[SculptBrushRow] = []
        sculpt_float_rows: list[SculptFloatRow] = []
        sculpt_buttons: list[Button] = []
        if active_right is DockPanel.sculpt and not modal_open:
            sculpt_brush_rows, sculpt_float_rows, sculpt_buttons = update_sculpt_panel(
                self.ui, right_content_rect
            )

        inspector_buttons: list[Button] = []
        inspector_rows: list[ParamRowRects] = []
        vector_inspector_buttons: list[Button] = []
        vector_float_rows: list[VectorShapeFloatRow] = []
        vector_layer_buttons: list[Button] = []
        road_sections_layout: RoadSectionsInspectorLayout = (
            RoadSectionsInspectorLayout()
        )
        modifiers_buttons: list[Button] = []
        modifiers_rows: list[ModifierPanelRow] = []
        if active_right is DockPanel.inspector and not modal_open:
            if vector_shape is not None:
                vector_inspector_buttons, vector_float_rows = (
                    update_vector_shape_inspector(
                        self.scene.nodes,
                        self.application.heightmap_catalog,
                        self.ui,
                        right_content_rect,
                        vector_shape,
                        heightfield_gpu=self.scene.heightfield_gpu,
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
            elif vector_layer is not None:
                vector_layer_buttons = update_vector_layer_inspector(
                    self.scene.nodes,
                    self.application.heightmap_catalog,
                    self.ui,
                    right_content_rect,
                    vector_layer,
                    heightfield_gpu=self.scene.heightfield_gpu,
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
                            right_content_rect,
                            vector_shape,
                            heightfield_gpu=self.scene.heightfield_gpu,
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
            elif road_sections_layer is not None:
                road_sections_layout = update_road_sections_inspector(
                    self.scene.nodes,
                    self.ui,
                    right_content_rect,
                    road_sections_layer,
                    journal=self.application.journal,
                    road_parts=self.application.road_parts,
                    scene_context=self.scene,
                    heightmap_catalog=self.application.heightmap_catalog,
                )
            elif group is not None:
                group_id: str = group.id
                inspector_buttons, inspector_rows = update_inspector(
                    self.scene.nodes,
                    self.ui,
                    right_content_rect,
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
        elif (
            active_right is DockPanel.modifiers
            and group is not None
            and group.generator is not None
            and not modal_open
        ):
            modifiers_group_id: str = group.id
            modifiers_buttons, modifiers_rows = update_modifiers_panel(
                self.scene.nodes,
                self.ui,
                right_content_rect,
                group,
                journal=self.application.journal,
                heightmap_catalog=self.application.heightmap_catalog,
            )
            modifiers_refreshed: Node | None = self.scene.nodes.nodes.get(
                modifiers_group_id
            )
            if modifiers_refreshed is None or (
                modifiers_refreshed.generator is None
                and modifiers_refreshed.heightfield is None
            ):
                group = None
            else:
                group = modifiers_refreshed

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
                    self.ui.road_sections_import_target = None
                    self.ui.tb_import_target = None
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
                        case FileBrowserPurpose.export_heightmap:
                            apply_export_heightmap_path(self, chosen)
                        case FileBrowserPurpose.import_road_shp:
                            apply_import_road_shp_path(self, chosen)
                        case FileBrowserPurpose.import_heightfield_albedo:
                            apply_import_heightfield_albedo_path(self, chosen)
                        case FileBrowserPurpose.export_tb_roads:
                            apply_export_tb_roads_path(self, chosen)
                        case FileBrowserPurpose.import_tb_objects:
                            apply_import_tb_objects_path(self, chosen)
                case None:
                    pass

        picker_rows: list[ListPickerRow] = []
        picker_buttons: list[Button] = []
        picker_window: Rectangle | None = None
        if self.ui.list_picker is not None:
            picker_result: ListPickerFrameResult
            picker_result, picker_rows, picker_buttons, picker_window = (
                update_list_picker(
                    self.ui.list_picker,
                    get_screen_width(),
                    get_screen_height(),
                )
            )
            match picker_result:
                case ListPickerCancel.cancelled:
                    self.ui.list_picker = None
                    self.ui.status = "Cancelled"
                case str() as chosen:
                    purpose_picker: ListPickerPurpose = self.ui.list_picker.purpose
                    target_id: str | None = self.ui.list_picker.target_id
                    self.ui.list_picker = None
                    match purpose_picker:
                        case ListPickerPurpose.road_kit:
                            if target_id is None:
                                self.ui.status = "Kit target missing"
                            else:
                                apply_road_kit_choice(
                                    self.scene.nodes,
                                    self.ui,
                                    target_id,
                                    chosen,
                                    self.application.journal,
                                    road_parts=self.application.road_parts,
                                    scene_context=self.scene,
                                    heightmap_catalog=self.application.heightmap_catalog,
                                )
                        case ListPickerPurpose.road_move_usage:
                            if target_id is None:
                                self.ui.status = "Move source missing"
                            elif chosen == list_picker_new_item:
                                begin_move_usage_name(self.ui, target_id)
                            else:
                                apply_move_polylines_to_usage(
                                    self.scene.nodes,
                                    self.ui,
                                    target_id,
                                    chosen,
                                    self.application.journal,
                                    road_parts=self.application.road_parts,
                                    scene_context=self.scene,
                                    heightmap_catalog=self.application.heightmap_catalog,
                                )
                        case ListPickerPurpose.vector_brush:
                            if target_id is None:
                                self.ui.status = "Brush target missing"
                            else:
                                apply_vector_brush_choice(
                                    self.scene.nodes,
                                    self.application.heightmap_catalog,
                                    self.ui,
                                    target_id,
                                    chosen,
                                    heightfield_gpu=self.scene.heightfield_gpu,
                                )
                case None:
                    pass

        palette_swatches: list[ColorPaletteSwatch] = []
        palette_window: Rectangle | None = None
        if self.ui.color_palette is not None:
            palette_result: ColorPaletteFrameResult
            palette_result, palette_swatches, palette_window = (
                update_color_palette_popup(
                    self.ui.color_palette,
                    get_screen_width(),
                    get_screen_height(),
                )
            )
            match palette_result:
                case ColorPaletteCancel():
                    self.ui.color_palette = None
                case ColorPaletteClear():
                    owner_id: str = self.ui.color_palette.owner_id
                    self.ui.color_palette = None
                    set_node_color_index(self.scene.nodes, owner_id, None)
                case int() as chosen_index:
                    owner_pick: str = self.ui.color_palette.owner_id
                    self.ui.color_palette = None
                    set_node_color_index(
                        self.scene.nodes, owner_pick, chosen_index
                    )
                case None:
                    pass

        mouse: Vector2 = get_mouse_position()
        ui_over: bool = modal_open or (
            is_point_in_rect(mouse.x, mouse.y, layout.menu)
            or (
                len(left_tabs) > 0
                and is_point_in_rect(mouse.x, mouse.y, layout.left_dock)
            )
            or (
                len(right_tabs) > 0
                and is_point_in_rect(mouse.x, mouse.y, layout.right_dock)
            )
            or is_point_in_rect(mouse.x, mouse.y, layout.status)
            or self.ui.dock_resize.side is not None
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
                    stamped_hf: Heightfield | None = apply_sculpt_stamp(
                        self.application.heightmap_catalog,
                        heightfield_group.heightfield,
                        self.sculpt_stroke,
                        self.ui.sculpt_brush_kind,
                        hit.u,
                        hit.v,
                        self.ui.sculpt_radius,
                        self.ui.sculpt_intensity,
                    )
                    if stamped_hf is not None:
                        self.scene.nodes.add_nodes(
                            [
                                replace(
                                    heightfield_group,
                                    heightfield=stamped_hf,
                                )
                            ]
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
                live_group: Node | None = self.scene.nodes.nodes.get(
                    heightfield_group.id
                )
                live_hf: Heightfield | None = (
                    None
                    if live_group is None
                    else live_group.heightfield
                )
                if live_hf is not None and live_group is not None:
                    stamped_hf = apply_sculpt_stamp(
                        self.application.heightmap_catalog,
                        live_hf,
                        self.sculpt_stroke,
                        self.ui.sculpt_brush_kind,
                        hit.u,
                        hit.v,
                        self.ui.sculpt_radius,
                        self.ui.sculpt_intensity,
                    )
                    if stamped_hf is not None:
                        self.scene.nodes.add_nodes(
                            [
                                replace(
                                    live_group,
                                    heightfield=stamped_hf,
                                )
                            ]
                        )
            if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT):
                self.sculpt_stroke = None
        elif self.sculpt_stroke is not None:
            self.sculpt_stroke = None
        self.scene.handle_input(
            layout.viewport,
            ui_over,
            self.application.heightmap_catalog,
            text_focused=(
                self.ui.focus is not None
                or self.ui.file_browser is not None
                or self.ui.list_picker is not None
                or self.ui.color_palette is not None
            ),
            ui=self.ui,
            block_left_click=sculpting and not ui_over,
            road_parts=self.application.road_parts,
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
            self.ui,
            sculpt_preview_center=preview_center,
            sculpt_preview_radius=preview_radius,
            road_parts=self.application.road_parts,
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
        if left_tabs:
            draw_tab_bar(self.font, layout.left_dock, left_tabs, active_left)
        if right_tabs:
            draw_tab_bar(self.font, layout.right_dock, right_tabs, active_right)
        if active_left is DockPanel.tools:
            draw_button_stack(self.font, left_content, panel_buttons)
        if active_left is DockPanel.outliner:
            draw_outliner(
                self.font, left_content, self.ui, outliner_rows, outliner_buttons
            )
        if active_right is DockPanel.meshes:
            draw_mesh_panel(self.font, right_content_rect, mesh_rows, mesh_buttons)
        if active_right is DockPanel.height_layers:
            draw_height_layers_panel(
                self.font,
                right_content_rect,
                self.ui,
                heightfield_group,
                height_layer_rows,
                height_layer_float_rows,
                height_layer_buttons,
                height_layers_blend_label,
                height_layers_empty,
            )
        if active_right is DockPanel.sculpt:
            draw_sculpt_panel(
                self.font,
                right_content_rect,
                self.ui,
                sculpt_brush_rows,
                sculpt_float_rows,
                sculpt_buttons,
            )
        if active_right is DockPanel.inspector:
            if vector_shape is not None:
                draw_vector_shape_inspector(
                    self.font,
                    right_content_rect,
                    self.ui,
                    vector_shape,
                    vector_inspector_buttons,
                    vector_float_rows,
                )
            elif vector_layer is not None:
                draw_vector_layer_inspector(
                    self.font,
                    right_content_rect,
                    vector_layer,
                    vector_layer_buttons,
                )
            elif road_sections_layer is not None:
                draw_road_sections_inspector(
                    self.font,
                    right_content_rect,
                    road_sections_layer,
                    self.ui,
                    road_sections_layout,
                )
            elif group is not None:
                draw_inspector(
                    self.font,
                    right_content_rect,
                    self.ui,
                    group,
                    inspector_buttons,
                    inspector_rows,
                )
        elif (
            active_right is DockPanel.modifiers
            and group is not None
            and group.generator is not None
        ):
            draw_modifiers_panel(
                self.font,
                right_content_rect,
                self.ui,
                group,
                modifiers_buttons,
                modifiers_rows,
                self.scene.nodes,
            )
        orbit_target: Vector3 | None = self.scene.orbit.target
        assert orbit_target is not None
        draw_status_bar(
            self.font,
            layout.status,
            self.ui.status,
            fps=get_fps(),
            frame_ms=get_frame_time() * 1000.0,
            look=(
                f"look {orbit_target.x:.1f} {orbit_target.z:.1f}"
                f"  d {self.scene.orbit.distance:.0f}"
            ),
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
        if self.ui.list_picker is not None and picker_window is not None:
            draw_list_picker(
                self.font,
                self.ui.list_picker,
                picker_rows,
                picker_buttons,
                picker_window,
                get_screen_width(),
                get_screen_height(),
            )
        if self.ui.color_palette is not None and palette_window is not None:
            draw_color_palette_popup(
                self.font,
                self.ui.color_palette,
                palette_swatches,
                palette_window,
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
