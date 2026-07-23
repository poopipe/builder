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
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.theme import COLOUR_BG, FONT_SIZE, SIDE_PANEL_WIDTH
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    Button,
    LayoutRects,
    compute_layout,
    draw_button_stack,
    draw_menu_bar,
    draw_status_bar,
    is_point_in_rect,
    layout_menu_bar_buttons,
    layout_stack_buttons,
    update_buttons,
    update_clipped_buttons,
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
        panel_width: int = SIDE_PANEL_WIDTH if self.ui.side_panel_open else 0
        layout: LayoutRects = compute_layout(
            get_screen_width(),
            get_screen_height(),
            panel_width=panel_width,
        )

        menu_buttons: list[Button] = layout_menu_bar_buttons(
            self.font,
            layout.menu,
            self.command_button_items(MENU_COMMANDS),
        )
        update_buttons(menu_buttons)

        panel_buttons: list[Button] = []
        if self.ui.side_panel_open:
            panel_buttons = layout_stack_buttons(
                layout.panel,
                self.command_button_items(PANEL_COMMANDS),
            )
            update_clipped_buttons(panel_buttons, layout.panel)

        mouse: Vector2 = get_mouse_position()
        ui_over: bool = (
            is_point_in_rect(mouse.x, mouse.y, layout.menu)
            or (
                self.ui.side_panel_open
                and is_point_in_rect(mouse.x, mouse.y, layout.panel)
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
