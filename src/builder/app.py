"""Application session and main window loop."""

from __future__ import annotations

from pyray import (
    ConfigFlags,
    Font,
    KeyboardKey,
    begin_drawing,
    clear_background,
    close_window,
    end_drawing,
    get_mouse_position,
    get_screen_height,
    get_screen_width,
    init_window,
    set_config_flags,
    set_exit_key,
    set_target_fps,
    window_should_close,
)

from builder.application_state import ApplicationState
from builder.commands.builtin import (
    MENU_COMMANDS,
    PANEL_COMMANDS,
    builtin_commands,
)
from builder.commands.registry import bind_command, register_commands
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.theme import COLOUR_BG, FONT_SIZE, SIDE_PANEL_WIDTH
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    MenuItem,
    StackButton,
    compute_layout,
    draw_button_stack,
    draw_menu_bar,
    draw_status_bar,
    is_point_in_rect,
)
from builder.view.viewport import Viewport


class Application:
    """Owns session state and runs the main loop.

    Structurally satisfies ``CommandContext`` (application, scene, ui).
    Construct only after the raylib window exists so ``scene`` and ``font``
    can be created up front.
    """

    def __init__(self, scene: Viewport, font: Font) -> None:
        self.application = ApplicationState()
        self.scene = scene
        self.ui = UiState(status=self.application.importer.status_message())
        self.font = font
        self.commands = register_commands(builtin_commands())

    def menu_items(self) -> list[MenuItem]:
        return [
            MenuItem(command.label, bind_command(self, command))
            for command in MENU_COMMANDS
        ]

    def panel_buttons(self) -> list[StackButton]:
        return [
            StackButton(command.label, bind_command(self, command))
            for command in PANEL_COMMANDS
        ]

    def frame(self) -> None:
        panel_width = SIDE_PANEL_WIDTH if self.ui.side_panel_open else 0
        layout = compute_layout(
            get_screen_width(),
            get_screen_height(),
            panel_width=panel_width,
        )
        mouse = get_mouse_position()
        ui_over = (
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
        draw_menu_bar(self.font, layout.menu, self.menu_items())
        if self.ui.side_panel_open:
            draw_button_stack(self.font, layout.panel, self.panel_buttons())
        draw_status_bar(self.font, layout.status, self.ui.status)
        end_drawing()

    def run(self) -> None:
        """Pump frames until quit."""
        while not window_should_close() and not self.application.should_close:
            self.frame()


def open_window() -> None:
    """Create the raylib window (required before GPU resources)."""
    set_config_flags(
        ConfigFlags.FLAG_WINDOW_RESIZABLE | ConfigFlags.FLAG_MSAA_4X_HINT
    )
    init_window(1280, 720, "Builder")
    set_target_fps(60)
    set_exit_key(KeyboardKey.KEY_ESCAPE)


def run_application() -> None:
    """Open the window, build the application, and run until quit."""
    open_window()
    font = load_app_font(FONT_SIZE)
    scene = Viewport()
    app = Application(scene=scene, font=font)
    try:
        app.run()
    finally:
        scene.unload()
        unload_app_font(font)
        close_window()
