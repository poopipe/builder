"""Application session and main window loop."""

from __future__ import annotations

from collections.abc import Callable

from pyray import (
    ConfigFlags,
    Font,
    KeyboardKey,
    Vector2,
    begin_drawing,
    clear_background,
    close_window,
    end_drawing,
    get_mouse_position,
    get_screen_height,
    get_screen_width,
    init_window,
    is_window_ready,
    set_config_flags,
    set_exit_key,
    set_target_fps,
    window_should_close,
)

from builder.application.application_state import ApplicationState
from builder.commands.commands_types import CommandItem
from builder.commands.menus import MENU_COMMANDS, PANEL_COMMANDS, all_commands
from builder.commands.registry import bind_entry, register_commands
from builder.generators.regenerate import selected_parametric_group
from builder.scene.scene_types import Node
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.inspector import (
    ParamRowRects,
    apply_inspector_exit_key,
    draw_inspector,
    sync_inspector_focus,
    update_inspector,
)
from builder.ui.theme import (
    BUTTON_GAP,
    BUTTON_HEIGHT,
    BUTTON_MIN_WIDTH,
    COLOUR_BG,
    FONT_SIZE,
    INSPECTOR_PANEL_WIDTH,
    PAD,
    SIDE_PANEL_WIDTH,
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


class Application:
    """Owns session state and runs the main loop.

    Structurally satisfies ``CommandContext`` (application, scene, ui).
    Construct only after the raylib window exists so ``scene`` and ``font``
    can be created up front.
    """

    def __init__(self, scene: Viewport, font: Font) -> None:
        self.application: ApplicationState = ApplicationState()
        self.scene: Viewport = scene
        self.ui: UiState = UiState(status=self.application.importer.status_message())
        self.font: Font = font
        self.commands = register_commands(all_commands())

    def command_button_items(
        self, entries: tuple[CommandItem, ...]
    ) -> list[tuple[str, Callable[[], None]]]:
        items: list[tuple[str, Callable[[], None]]] = []
        entry: CommandItem
        for entry in entries:
            items.append((entry.command.label, bind_entry(self, entry)))
        return items

    def frame(self) -> None:
        """define the UI layout"""
        group: Node | None = selected_parametric_group(self.scene.nodes)
        sync_inspector_focus(self.ui, group)
        apply_inspector_exit_key(self.ui)

        panel_width: int = SIDE_PANEL_WIDTH if self.ui.side_panel_open else 0
        inspector_width: int = INSPECTOR_PANEL_WIDTH if group is not None else 0
        layout: LayoutRects = compute_layout(
            get_screen_width(),
            get_screen_height(),
            panel_width=panel_width,
            inspector_width=inspector_width,
        )

        menu_buttons: list[Button] = layout_buttons_horizontal(
            self.font,
            layout.menu,
            self.command_button_items(MENU_COMMANDS),
            pad=float(PAD),
            gap=float(BUTTON_GAP),
            button_height=float(BUTTON_HEIGHT),
            min_width=float(BUTTON_MIN_WIDTH),
            font_size=float(FONT_SIZE),
        )
        update_buttons(menu_buttons, layout.menu)
        """ 
            buttons run commands 
            commands are listed in consts (eg. PANEL_COMMANDS)
            executable code for command lives in a file per context (eg. view.py)
            CommandContext is used to give the command access to the app state 
            (application, scene, ui) 
        """

        panel_buttons: list[Button] = []
        if self.ui.side_panel_open:
            panel_buttons = layout_buttons_vertical(
                layout.panel,
                self.command_button_items(PANEL_COMMANDS),
                pad=float(PAD),
                gap=float(BUTTON_GAP),
                button_height=float(BUTTON_HEIGHT),
            )
            update_buttons(panel_buttons, layout.panel)

        inspector_buttons: list[Button] = []
        inspector_rows: list[ParamRowRects] = []
        if group is not None:
            group_id: str = group.id
            inspector_buttons, inspector_rows = update_inspector(
                self.scene.nodes,
                self.ui,
                layout.inspector,
                group,
            )
            # steppers/bake replace the group node; re-read the same id so the
            # drawn group stays consistent with rows even if selection changed
            refreshed: Node | None = self.scene.nodes.nodes.get(group_id)
            group = refreshed if refreshed is not None and refreshed.generator is not None else None

        mouse: Vector2 = get_mouse_position()
        ui_over: bool = (
            is_point_in_rect(mouse.x, mouse.y, layout.menu)
            or (
                self.ui.side_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.panel)
            )
            or (
                group is not None
                and is_point_in_rect(mouse.x, mouse.y, layout.inspector)
            )
            or is_point_in_rect(mouse.x, mouse.y, layout.status)
        )
        self.scene.handle_input(layout.viewport, ui_over)

        begin_drawing()
        clear_background(COLOUR_BG)
        self.scene.draw(layout.viewport)
        draw_menu_bar(self.font, layout.menu, menu_buttons)
        if self.ui.side_panel_open:
            draw_button_stack(self.font, layout.panel, panel_buttons)
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
        end_drawing()

    def run(self) -> None:
        while not window_should_close() and not self.application.should_close:
            self.frame()

    def shutdown(self) -> None:
        """Release GPU resources owned by this session."""
        self.scene.unload()
        unload_app_font(self.font)


def open_window() -> None:
    """Create the raylib window (required before GPU resources)."""
    set_config_flags(ConfigFlags.FLAG_WINDOW_RESIZABLE | ConfigFlags.FLAG_MSAA_4X_HINT)
    init_window(1280, 720, "Builder")
    if not is_window_ready():
        raise RuntimeError("failed to create application window")
    set_target_fps(60)
    set_exit_key(KeyboardKey.KEY_ESCAPE)


def run_application() -> None:
    """Open the window, build the application, run until quit, then tear down."""
    open_window()
    font: Font = load_app_font(FONT_SIZE)

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
