"""inspector for road sections layer: import shapefile"""

from __future__ import annotations

from pyray import (
    Font,
    Rectangle,
    Vector2,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
)

from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.file_browser import (
    FileBrowserPurpose,
    make_road_shp_browser,
    remembered_browser_directory,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_panel,
    ui_color_text,
    ui_font_size,
    ui_pad,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import Button, draw_button, update_buttons


def selected_road_sections_layer(scene: Scene) -> Node | None:
    """sole selected road sections layer, or layer owning selected shp/usage"""
    if len(scene.selected_ids) != 1:
        return None
    node: Node | None = scene.nodes.get(scene.selected_ids[0])
    if node is None:
        return None
    current: Node | None = node
    while current is not None:
        if current.road_sections_layer is not None:
            return current
        if current.parent_id is None:
            return None
        current = scene.nodes.get(current.parent_id)
    return None


def update_road_sections_inspector(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    layer_node: Node,
) -> list[Button]:
    """layout/input: import shapefile button"""
    _ = scene
    if layer_node.road_sections_layer is None:
        return []
    buttons: list[Button] = []
    x: float = area.x + ui_pad
    y: float = area.y + ui_pad + float(ui_font_size) + ui_button_gap
    inner_w: float = area.width - 2.0 * ui_pad
    row_h: float = float(ui_button_height)

    def open_import() -> None:
        ui.clear_focus()
        ui.road_sections_import_target = layer_node.id
        ui.file_browser = make_road_shp_browser(
            remembered_browser_directory(
                ui.browser_directories,
                FileBrowserPurpose.import_road_shp,
                None,
            )
        )
        ui.status = "Import road shapefile…"

    buttons.append(
        Button(
            label="Import SHP…",
            on_click=open_import,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    update_buttons(buttons, area)
    return buttons


def draw_road_sections_inspector(
    font: Font,
    area: Rectangle,
    layer_node: Node,
    buttons: list[Button],
) -> None:
    """draw road sections import controls"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(area, 1.0, ui_color_border)
    title: str = layer_node.name if layer_node.name else "Road Sections"
    draw_text_ex(
        font,
        title,
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        1.0,
        ui_color_text,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)
