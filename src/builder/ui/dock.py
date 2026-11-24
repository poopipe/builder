"""tabbed left/right docks and horizontal resize"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    draw_rectangle_rec,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    measure_text_ex,
)

from builder.ui.theme import (
    ui_color_border,
    ui_color_button,
    ui_color_button_hover,
    ui_color_button_press,
    ui_color_panel,
    ui_color_text,
    ui_dock_max_width,
    ui_dock_min_width,
    ui_font_size,
    ui_left_dock_default_width,
    ui_pad,
    ui_right_dock_default_width,
    ui_splitter_hit_pad,
    ui_tab_bar_height,
)
from builder.ui.widgets import is_point_in_rect


class DockPanel(IntEnum):
    """panel that can occupy a dock tab"""

    tools = 1
    outliner = 2
    inspector = 3
    modifiers = 4
    meshes = 5
    height_layers = 6
    sculpt = 7


class DockSide(IntEnum):
    """which dock column is being resized"""

    left = 1
    right = 2


@dataclass(frozen=True)
class RightDockContent:
    """selection-driven right-dock tabs"""

    show_inspector: bool
    show_modifiers: bool


_dock_panel_labels: dict[DockPanel, str] = {
    DockPanel.tools: "Tools",
    DockPanel.outliner: "Outliner",
    DockPanel.inspector: "Properties",
    DockPanel.modifiers: "Modifiers",
    DockPanel.meshes: "Meshes",
    DockPanel.height_layers: "Layers",
    DockPanel.sculpt: "Sculpt",
}


def dock_panel_label(panel: DockPanel) -> str:
    """short tab label for a dock panel"""
    return _dock_panel_labels[panel]


def clamp_dock_width(width: float) -> float:
    """clamp a dock width to theme min/max"""
    return max(ui_dock_min_width, min(ui_dock_max_width, float(width)))


def default_left_dock_width() -> float:
    """default left dock width from theme"""
    return clamp_dock_width(ui_left_dock_default_width)


def default_right_dock_width() -> float:
    """default right dock width from theme"""
    return clamp_dock_width(ui_right_dock_default_width)


def available_left_tabs(*, side_panel_open: bool, outliner_open: bool) -> list[DockPanel]:
    """left-dock tabs present for the current open flags"""
    tabs: list[DockPanel] = []
    if side_panel_open:
        tabs.append(DockPanel.tools)
    if outliner_open:
        tabs.append(DockPanel.outliner)
    return tabs


def available_right_tabs(
    *,
    meshes_panel_open: bool,
    height_layers_panel_open: bool,
    sculpt_panel_open: bool,
    content: RightDockContent,
) -> list[DockPanel]:
    """right-dock tabs present for open flags and selection content"""
    tabs: list[DockPanel] = []
    if content.show_inspector:
        tabs.append(DockPanel.inspector)
    if content.show_modifiers:
        tabs.append(DockPanel.modifiers)
    if meshes_panel_open:
        tabs.append(DockPanel.meshes)
    if height_layers_panel_open:
        tabs.append(DockPanel.height_layers)
    if sculpt_panel_open:
        tabs.append(DockPanel.sculpt)
    return tabs


def resolve_active_tab(
    current: DockPanel,
    available: Sequence[DockPanel],
    prefer: DockPanel | None = None,
) -> DockPanel | None:
    """keep current if available, else prefer, else first"""
    if not available:
        return None
    if current in available:
        return current
    if prefer is not None and prefer in available:
        return prefer
    return available[0]


def tab_bar_rect(dock: Rectangle) -> Rectangle:
    """strip at the top of a dock for tabs"""
    if dock.width <= 0.0 or dock.height <= 0.0:
        return Rectangle(dock.x, dock.y, 0.0, 0.0)
    height: float = min(float(ui_tab_bar_height), dock.height)
    return Rectangle(dock.x, dock.y, dock.width, height)


def update_tab_bar(
    dock: Rectangle,
    tabs: Sequence[DockPanel],
    active: DockPanel | None,
) -> DockPanel | None:
    """click a tab to select it; returns the active tab"""
    if not tabs or dock.width <= 0.0:
        return None
    bar: Rectangle = tab_bar_rect(dock)
    if bar.height <= 0.0:
        return active if active in tabs else tabs[0]
    mouse: Vector2 = get_mouse_position()
    if not is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        return active if active in tabs else tabs[0]
    if not is_point_in_rect(mouse.x, mouse.y, bar):
        return active if active in tabs else tabs[0]
    tab_w: float = bar.width / float(len(tabs))
    index: int = int((mouse.x - bar.x) / tab_w)
    index = max(0, min(len(tabs) - 1, index))
    return tabs[index]


def draw_tab_bar(
    font: Font,
    dock: Rectangle,
    tabs: Sequence[DockPanel],
    active: DockPanel | None,
) -> None:
    """draw dock tab strip"""
    if not tabs or dock.width <= 0.0:
        return
    bar: Rectangle = tab_bar_rect(dock)
    if bar.height <= 0.0:
        return
    draw_rectangle_rec(bar, ui_color_panel)
    mouse: Vector2 = get_mouse_position()
    tab_w: float = bar.width / float(len(tabs))
    index: int
    panel: DockPanel
    for index, panel in enumerate(tabs):
        tab_rect: Rectangle = Rectangle(
            bar.x + float(index) * tab_w,
            bar.y,
            tab_w,
            bar.height,
        )
        hovered: bool = is_point_in_rect(mouse.x, mouse.y, tab_rect)
        selected: bool = panel is active
        fill = ui_color_button_press if selected else (
            ui_color_button_hover if hovered else ui_color_button
        )
        draw_rectangle_rec(tab_rect, fill)
        label: str = dock_panel_label(panel)
        size: float = float(ui_font_size - 2)
        text_w: float = measure_text_ex(font, label, size, 0).x
        draw_text_ex(
            font,
            label,
            Vector2(
                tab_rect.x + max(float(ui_pad) * 0.5, (tab_w - text_w) * 0.5),
                tab_rect.y + (tab_rect.height - size) * 0.5,
            ),
            size,
            0,
            ui_color_text,
        )
    draw_rectangle_rec(
        Rectangle(bar.x, bar.y + bar.height - 1.0, bar.width, 1.0),
        ui_color_border,
    )


@dataclass
class DockResizeState:
    """session-only dock edge drag"""

    side: DockSide | None = None
    start_x: float = 0.0
    start_width: float = 0.0


def splitter_hit_rect(side: DockSide, dock: Rectangle) -> Rectangle:
    """hit target on the viewport-facing dock edge"""
    pad: float = float(ui_splitter_hit_pad)
    if dock.width <= 0.0:
        return Rectangle(0.0, 0.0, 0.0, 0.0)
    if side is DockSide.left:
        return Rectangle(
            dock.x + dock.width - pad,
            dock.y,
            pad * 2.0,
            dock.height,
        )
    return Rectangle(dock.x - pad, dock.y, pad * 2.0, dock.height)


def update_dock_resize(
    resize: DockResizeState,
    *,
    left_dock: Rectangle,
    right_dock: Rectangle,
    left_width: float,
    right_width: float,
) -> tuple[float, float]:
    """drag viewport-facing dock edges; returns updated widths"""
    mouse: Vector2 = get_mouse_position()
    left_w: float = clamp_dock_width(left_width) if left_dock.width > 0.0 else left_width
    right_w: float = (
        clamp_dock_width(right_width) if right_dock.width > 0.0 else right_width
    )
    if resize.side is not None:
        if is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT) or not is_mouse_button_down(
            MouseButton.MOUSE_BUTTON_LEFT
        ):
            resize.side = None
        else:
            delta: float
            if resize.side is DockSide.left:
                delta = mouse.x - resize.start_x
                left_w = clamp_dock_width(resize.start_width + delta)
            else:
                delta = resize.start_x - mouse.x
                right_w = clamp_dock_width(resize.start_width + delta)
            return left_w, right_w
    if is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        if left_dock.width > 0.0 and is_point_in_rect(
            mouse.x, mouse.y, splitter_hit_rect(DockSide.left, left_dock)
        ):
            resize.side = DockSide.left
            resize.start_x = mouse.x
            resize.start_width = left_w
        elif right_dock.width > 0.0 and is_point_in_rect(
            mouse.x, mouse.y, splitter_hit_rect(DockSide.right, right_dock)
        ):
            resize.side = DockSide.right
            resize.start_x = mouse.x
            resize.start_width = right_w
    return left_w, right_w
