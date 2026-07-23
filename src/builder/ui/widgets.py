"""Immediate-mode UI widgets: menu bar and clipped button stack."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    begin_scissor_mode,
    draw_line,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    end_scissor_mode,
    get_mouse_position,
    is_mouse_button_down,
    is_mouse_button_released,
    measure_text_ex,
)

from builder.ui.theme import (
    BUTTON_GAP,
    BUTTON_HEIGHT,
    COLOUR_BORDER,
    COLOUR_BUTTON,
    COLOUR_BUTTON_HOVER,
    COLOUR_BUTTON_PRESS,
    COLOUR_MENU,
    COLOUR_PANEL,
    COLOUR_STATUS,
    COLOUR_TEXT,
    FONT_SIZE,
    MENU_BAR_HEIGHT,
    PAD,
    SIDE_PANEL_WIDTH,
    STATUS_BAR_HEIGHT,
)


@dataclass
class MenuItem:
    """A single top-bar menu action."""

    label: str
    on_click: Callable[[], None]


@dataclass
class StackButton:
    """A single context-panel button."""

    label: str
    on_click: Callable[[], None]


@dataclass
class LayoutRects:
    """Window regions for the current frame."""

    menu: Rectangle
    panel: Rectangle
    viewport: Rectangle
    status: Rectangle


def compute_layout(
    width: int,
    height: int,
    panel_width: int = SIDE_PANEL_WIDTH,
) -> LayoutRects:
    """Split the window into menu, side panel, viewport, and status bar."""
    menu_h = MENU_BAR_HEIGHT
    status_h = STATUS_BAR_HEIGHT
    panel_w = max(0, panel_width)
    body_y = menu_h
    body_h = max(0, height - menu_h - status_h)
    return LayoutRects(
        menu=Rectangle(0, 0, float(width), float(menu_h)),
        panel=Rectangle(0, float(body_y), float(panel_w), float(body_h)),
        viewport=Rectangle(
            float(panel_w),
            float(body_y),
            float(max(0, width - panel_w)),
            float(body_h),
        ),
        status=Rectangle(
            0,
            float(height - status_h),
            float(width),
            float(status_h),
        ),
    )


def is_point_in_rect(x: float, y: float, rect: Rectangle) -> bool:
    """Return True if (x, y) lies inside rect."""
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def is_button_clicked(
    font: Font,
    rect: Rectangle,
    label: str,
    mouse: Vector2,
    mouse_pressed: bool,
) -> bool:
    hovered = is_point_in_rect(mouse.x, mouse.y, rect)
    pressed = hovered and mouse_pressed
    if pressed:
        colour = COLOUR_BUTTON_PRESS
    elif hovered:
        colour = COLOUR_BUTTON_HOVER
    else:
        colour = COLOUR_BUTTON

    draw_rectangle_rec(rect, colour)
    draw_rectangle_lines_ex(rect, 1, COLOUR_BORDER)

    text_size = FONT_SIZE
    text_w = measure_text_ex(font, label, float(text_size), 0).x
    text_h = float(text_size)
    tx = rect.x + (rect.width - text_w) * 0.5
    ty = rect.y + (rect.height - text_h) * 0.5
    draw_text_ex(
        font,
        label,
        Vector2(tx, ty),
        float(text_size),
        0,
        COLOUR_TEXT,
    )

    return hovered and is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)


def draw_menu_bar(
    font: Font,
    rect: Rectangle,
    items: Sequence[MenuItem],
) -> None:
    """Draw the horizontal menu bar and handle item clicks."""
    draw_rectangle_rec(rect, COLOUR_MENU)
    draw_line(
        int(rect.x),
        int(rect.y + rect.height - 1),
        int(rect.x + rect.width),
        int(rect.y + rect.height - 1),
        COLOUR_BORDER,
    )

    mouse = get_mouse_position()
    x = rect.x + PAD
    y = rect.y + (rect.height - BUTTON_HEIGHT) * 0.5
    for item in items:
        label_w = measure_text_ex(font, item.label, float(FONT_SIZE), 0).x
        bw = max(64.0, label_w + PAD * 2)
        btn = Rectangle(x, y, bw, float(BUTTON_HEIGHT))
        if is_button_clicked(
            font,
            btn,
            item.label,
            mouse,
            is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT),
        ):
            item.on_click()
        x += bw + BUTTON_GAP


def draw_button_stack(
    font: Font,
    rect: Rectangle,
    buttons: Sequence[StackButton],
) -> None:
    """Draw a vertically stacked, scissor-clipped button panel."""
    draw_rectangle_rec(rect, COLOUR_PANEL)
    draw_line(
        int(rect.x + rect.width - 1),
        int(rect.y),
        int(rect.x + rect.width - 1),
        int(rect.y + rect.height),
        COLOUR_BORDER,
    )

    mouse = get_mouse_position()
    over = is_point_in_rect(mouse.x, mouse.y, rect)

    begin_scissor_mode(
        int(rect.x),
        int(rect.y),
        int(rect.width),
        int(rect.height),
    )
    y = rect.y + PAD
    x = rect.x + PAD
    bw = rect.width - PAD * 2
    for button in buttons:
        btn = Rectangle(x, y, bw, float(BUTTON_HEIGHT))
        visible = btn.y + btn.height > rect.y and btn.y < rect.y + rect.height
        if visible and is_button_clicked(
            font,
            btn,
            button.label,
            mouse,
            is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT),
        ):
            if over:
                button.on_click()
        y += BUTTON_HEIGHT + BUTTON_GAP
    end_scissor_mode()


def draw_status_bar(font: Font, rect: Rectangle, text: str) -> None:
    """Draw a single-line status bar."""
    draw_rectangle_rec(rect, COLOUR_STATUS)
    draw_text_ex(
        font,
        text,
        Vector2(rect.x + PAD, rect.y + 4),
        float(FONT_SIZE - 2),
        0,
        COLOUR_TEXT,
    )
