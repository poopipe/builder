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
from builder.commands.builtin import register_builtin_commands
from builder.commands.commands_types import CommandId
from builder.commands.registry import CommandRegistry
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

MENU_COMMANDS: tuple[CommandId, ...] = (
    "open",
    "quit",
    "about",
    "toggle_side_panel",
)
PANEL_COMMANDS: tuple[CommandId, ...] = (
    "import_mesh",
    "toggle_grid",
    "focus_camera",
    "nudge_placeholder",
    "clear_selection",
    "place_example",
)


class Application:
    """Owns session state and runs the main loop.

    Structurally satisfies ``CommandContext`` (application, scene, ui).
    """

    def __init__(self) -> None:
        self.application = ApplicationState()
        self.scene: Viewport | None = None
        self.ui = UiState()
        self.font: Font | None = None
        self.commands = CommandRegistry()

    def menu_items(self) -> list[MenuItem]:
        items: list[MenuItem] = []
        for command_id in MENU_COMMANDS:
            command = self.commands.get(command_id)
            if command is None:
                continue
            items.append(MenuItem(command.label, self.commands.bind(self, command_id)))
        return items

    def panel_buttons(self) -> list[StackButton]:
        buttons: list[StackButton] = []
        for command_id in PANEL_COMMANDS:
            command = self.commands.get(command_id)
            if command is None:
                continue
            buttons.append(
                StackButton(command.label, self.commands.bind(self, command_id))
            )
        return buttons

    def frame(self) -> None:
        assert self.scene is not None
        assert self.font is not None

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
        """Create the window and run until quit."""
        register_builtin_commands(self.commands)

        set_config_flags(
            ConfigFlags.FLAG_WINDOW_RESIZABLE | ConfigFlags.FLAG_MSAA_4X_HINT
        )
        init_window(1280, 720, "Builder")
        set_target_fps(60)
        set_exit_key(KeyboardKey.KEY_ESCAPE)

        self.font = load_app_font(FONT_SIZE)
        self.scene = Viewport()
        self.ui.status = self.application.importer.status_message()

        while not window_should_close() and not self.application.should_close:
            self.frame()

        assert self.scene is not None
        assert self.font is not None
        self.scene.unload()
        unload_app_font(self.font)
        close_window()
