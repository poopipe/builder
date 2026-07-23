"""Application window loop, layout, and input routing."""

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

from builder.io.mesh_import import MeshImporter
from builder.ui.font import load_app_font, unload_app_font
from builder.ui.theme import COLOUR_BG, FONT_SIZE
from builder.ui.widgets import (
    MenuItem,
    StackButton,
    compute_layout,
    draw_button_stack,
    draw_menu_bar,
    draw_status_bar,
    point_in_rect,
)
from builder.view.viewport import Viewport


class App:
    """Top-level application orchestrator."""

    def __init__(self) -> None:
        self._should_close = False
        self.status = "Ready"
        self.importer = MeshImporter()
        self.viewport: Viewport | None = None
        self.font: Font | None = None

    def run(self) -> None:
        """Create the window and run until quit."""
        set_config_flags(
            ConfigFlags.FLAG_WINDOW_RESIZABLE | ConfigFlags.FLAG_MSAA_4X_HINT
        )
        init_window(1280, 720, "Builder")
        set_target_fps(60)
        set_exit_key(KeyboardKey.KEY_ESCAPE)

        self.font = load_app_font(FONT_SIZE)
        self.viewport = Viewport()
        self.status = self.importer.status_message()

        while not window_should_close() and not self._should_close:
            self._frame()

        assert self.viewport is not None
        assert self.font is not None
        self.viewport.unload()
        unload_app_font(self.font)
        close_window()

    def _quit(self) -> None:
        self._should_close = True

    def _open(self) -> None:
        self.status = "Open: not implemented"

    def _toggle_grid(self) -> None:
        assert self.viewport is not None
        self.viewport.toggle_grid()
        self.status = f"Grid {'on' if self.viewport.show_grid else 'off'}"

    def _about(self) -> None:
        self.status = "Builder framework — pyray / raylib 6"

    def _import_mesh(self) -> None:
        self.status = (
            "Import mesh: no file dialog yet. " + self.importer.status_message()
        )

    def _clear_selection(self) -> None:
        self.status = "Selection cleared (placeholder)"

    def _place_example(self) -> None:
        self.status = "Place: procedural placement not implemented yet"

    def _extra_context(self, index: int) -> None:
        self.status = f"Context action {index}"

    def _frame(self) -> None:
        assert self.viewport is not None
        assert self.font is not None

        width = get_screen_width()
        height = get_screen_height()
        layout = compute_layout(width, height)

        menu_items = [
            MenuItem("Open", self._open),
            MenuItem("Quit", self._quit),
            MenuItem("Grid", self._toggle_grid),
            MenuItem("About", self._about),
        ]

        # Extra stack buttons so clipping is obvious when the window is short.
        stack = [
            StackButton("Import mesh", self._import_mesh),
            StackButton("Focus camera", self.viewport.focus_origin),
            StackButton("Clear selection", self._clear_selection),
            StackButton("Place example", self._place_example),
        ]
        for i in range(1, 16):
            stack.append(StackButton(f"Tool {i}", lambda i=i: self._extra_context(i)))

        mouse = get_mouse_position()
        ui_over = (
            point_in_rect(mouse.x, mouse.y, layout.menu)
            or point_in_rect(mouse.x, mouse.y, layout.panel)
            or point_in_rect(mouse.x, mouse.y, layout.status)
        )
        self.viewport.handle_input(layout.viewport, ui_over)

        begin_drawing()
        clear_background(COLOUR_BG)
        self.viewport.draw(layout.viewport)
        draw_menu_bar(self.font, layout.menu, menu_items)
        draw_button_stack(self.font, layout.panel, stack)
        draw_status_bar(self.font, layout.status, self.status)
        end_drawing()
