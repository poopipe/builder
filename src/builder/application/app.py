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
    Rectangle,
    Vector2,
    begin_drawing,
    clear_background,
    close_window,
    end_drawing,
    ffi,
    get_mouse_position,
    get_screen_height,
    get_screen_width,
    init_window,
    is_file_dropped,
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
from builder.scene.scene_types import MeshId, Node
from builder.ui.file_browser import (
    BrowserRow,
    FileBrowserFrameResult,
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
from builder.ui.mesh_panel import MeshRow, draw_mesh_panel, update_mesh_panel
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
    ui_inspector_panel_width,
    ui_meshes_panel_width,
    ui_outliner_panel_width,
    ui_pad,
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
        self.ui: UiState = UiState(status=self.application.importer.status_message())
        self.font: Font = font
        self.commands: CommandRegistry = register_commands(all_commands())

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
                    import_mesh_from_path(self, path)
                except (OSError, RuntimeError, ValueError) as exc:
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
        sync_inspector_focus(self.ui, group)
        apply_text_exit_key(self.ui)

        panel_width: int = ui_side_panel_width if self.ui.side_panel_open else 0
        outliner_width: int = ui_outliner_panel_width if self.ui.outliner_open else 0
        meshes_width: int = ui_meshes_panel_width if self.ui.meshes_panel_open else 0
        inspector_width: int = ui_inspector_panel_width if group is not None else 0
        layout: LayoutRects = compute_layout(
            get_screen_width(),
            get_screen_height(),
            panel_width=panel_width,
            outliner_width=outliner_width,
            meshes_width=meshes_width,
            inspector_width=inspector_width,
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

        inspector_buttons: list[Button] = []
        inspector_rows: list[ParamRowRects] = []
        if group is not None and not browser_open:
            group_id: str = group.id
            inspector_buttons, inspector_rows = update_inspector(
                self.scene.nodes,
                self.ui,
                layout.inspector,
                group,
                self.application.mesh_catalog,
                self.application.active_mesh_id,
            )
            # steppers/bake replace the group node; re-read the same id so the
            # drawn group stays consistent with rows even if selection changed
            refreshed: Node | None = self.scene.nodes.nodes.get(group_id)
            group = (
                refreshed
                if refreshed is not None and refreshed.generator is not None
                else None
            )

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
            if browser_result == "cancelled":
                remember_browser_directory(self, self.ui.file_browser)
                self.ui.file_browser = None
                self.ui.status = "Cancelled"
            elif browser_result is not None:
                purpose: str = self.ui.file_browser.purpose
                remember_browser_directory(self, self.ui.file_browser)
                self.ui.file_browser = None
                if purpose == "open_scene":
                    apply_open_scene_path(self, browser_result)
                elif purpose == "save_scene":
                    apply_save_scene_path(self, browser_result)
                else:
                    apply_import_mesh_path(self, browser_result)

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
                group is not None
                and is_point_in_rect(mouse.x, mouse.y, layout.inspector)
            )
            or is_point_in_rect(mouse.x, mouse.y, layout.status)
        )
        self.scene.handle_input(layout.viewport, ui_over)

        begin_drawing()
        clear_background(ui_color_bg)
        self.scene.draw(layout.viewport)
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
        if self.ui.side_panel_open:
            draw_button_stack(self.font, layout.panel, panel_buttons)
        if self.ui.outliner_open:
            draw_outliner(
                self.font, layout.outliner, self.ui, outliner_rows, outliner_buttons
            )
        if self.ui.meshes_panel_open:
            draw_mesh_panel(self.font, layout.meshes, mesh_rows, mesh_buttons)
        if group is not None:
            draw_inspector(
                self.font,
                layout.inspector,
                self.ui,
                group,
                inspector_buttons,
                inspector_rows,
            )
        draw_status_bar(self.font, layout.status, self.ui.status)
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
            self.frame()

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
    except RuntimeError:
        unload_app_font(font)
        close_window()
        raise

    app: Application = Application(scene=scene, font=font)
    app.run()
    app.shutdown()
    close_window()
