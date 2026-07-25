"""UI widgets: layout, update, and draw as separate steps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pyray import (
    Color,
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
class Button:
    label: str
    on_click: Callable[[], None]
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False


@dataclass
class LayoutRects:
    """Window regions for the current frame."""

    menu: Rectangle
    panel: Rectangle
    viewport: Rectangle
    inspector: Rectangle
    status: Rectangle


def compute_layout(
    width: int,
    height: int,
    panel_width: int = SIDE_PANEL_WIDTH,
    inspector_width: int = 0,
) -> LayoutRects:
    """Split the window into menu, side panels, viewport, and status bar."""
    menu_h: int = MENU_BAR_HEIGHT
    status_h: int = STATUS_BAR_HEIGHT
    panel_w: int = max(0, panel_width)
    inspector_w: int = max(0, inspector_width)
    body_y: int = menu_h
    body_h: int = max(0, height - menu_h - status_h)
    viewport_w: int = max(0, width - panel_w - inspector_w)
    return LayoutRects(
        menu=Rectangle(0, 0, float(width), float(menu_h)),
        panel=Rectangle(0, float(body_y), float(panel_w), float(body_h)),
        viewport=Rectangle(
            float(panel_w),
            float(body_y),
            float(viewport_w),
            float(body_h),
        ),
        inspector=Rectangle(
            float(panel_w + viewport_w),
            float(body_y),
            float(inspector_w),
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


def is_button_in_clip(button: Button, clip: Rectangle) -> bool:
    """Return True if the button intersects the clip rectangle."""
    return (
        button.rect.y + button.rect.height > clip.y
        and button.rect.y < clip.y + clip.height
    )


def update_button(
    button: Button,
    mouse: Vector2,
    left_down: bool,
    left_released: bool,
    *,
    can_activate: bool = True,
) -> None:
    """Refresh hover/press state and fire ``on_click`` on release."""
    button.is_hovered = is_point_in_rect(mouse.x, mouse.y, button.rect)
    button.is_pressed = button.is_hovered and left_down
    if can_activate and button.is_hovered and left_released:
        button.on_click()


def update_buttons(buttons: Sequence[Button], clip: Rectangle) -> None:
    """update buttons, ignore clicks unless pointer and button are inside clip rectangle"""
    mouse: Vector2 = get_mouse_position()
    left_down: bool = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    pointer_in_clip: bool = is_point_in_rect(mouse.x, mouse.y, clip)
    button: Button
    for button in buttons:
        can_activate: bool = pointer_in_clip and is_button_in_clip(button, clip)
        update_button(
            button,
            mouse,
            left_down,
            left_released,
            can_activate=can_activate,
        )


def draw_button(button: Button, font: Font) -> None:
    """Draw a button from its current interaction state."""
    colour: Color
    if button.is_pressed:
        colour = COLOUR_BUTTON_PRESS
    elif button.is_hovered:
        colour = COLOUR_BUTTON_HOVER
    else:
        colour = COLOUR_BUTTON

    draw_rectangle_rec(button.rect, colour)
    draw_rectangle_lines_ex(button.rect, 1, COLOUR_BORDER)

    text_size: int = FONT_SIZE
    text_w: float = measure_text_ex(font, button.label, float(text_size), 0).x
    text_h: float = float(text_size)
    tx: float = button.rect.x + (button.rect.width - text_w) * 0.5
    ty: float = button.rect.y + (button.rect.height - text_h) * 0.5
    draw_text_ex(
        font,
        button.label,
        Vector2(tx, ty),
        float(text_size),
        0,
        COLOUR_TEXT,
    )


def layout_buttons_horizontal(
    font: Font,
    area: Rectangle,
    items: Sequence[tuple[str, Callable[[], None]]],
    *,
    pad: float,
    gap: float,
    button_height: float,
    min_width: float,
    font_size: float,
) -> list[Button]:
    """ place buttons left to right inside area """
    buttons: list[Button] = []
    x: float = area.x + pad
    y: float = area.y + (area.height - button_height) * 0.5
    label: str
    on_click: Callable[[], None]
    for label, on_click in items:
        label_w: float = measure_text_ex(font, label, font_size, 0).x
        bw: float = max(min_width, label_w + pad * 2)
        buttons.append(
            Button(label, on_click, Rectangle(x, y, bw, button_height))
        )
        x += bw + gap
    return buttons


def layout_buttons_vertical(
    area: Rectangle,
    items: Sequence[tuple[str, Callable[[], None]]],
    *,
    pad: float,
    gap: float,
    button_height: float,
) -> list[Button]:
    """ place buttons top to bottom inside area """
    buttons: list[Button] = []
    x: float = area.x + pad
    y: float = area.y + pad
    bw: float = area.width - pad * 2
    label: str
    on_click: Callable[[], None]
    for label, on_click in items:
        buttons.append(
            Button(label, on_click, Rectangle(x, y, bw, button_height))
        )
        y += button_height + gap
    return buttons


def draw_menu_bar(font: Font, bar: Rectangle, buttons: Sequence[Button]) -> None:
    """Draw the menu bar background and its buttons."""
    draw_rectangle_rec(bar, COLOUR_MENU)
    draw_line(
        int(bar.x),
        int(bar.y + bar.height - 1),
        int(bar.x + bar.width),
        int(bar.y + bar.height - 1),
        COLOUR_BORDER,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)


def draw_button_stack(font: Font, panel: Rectangle, buttons: Sequence[Button]) -> None:
    """Draw the side panel background and its clipped buttons."""
    draw_rectangle_rec(panel, COLOUR_PANEL)
    draw_line(
        int(panel.x + panel.width - 1),
        int(panel.y),
        int(panel.x + panel.width - 1),
        int(panel.y + panel.height),
        COLOUR_BORDER,
    )
    begin_scissor_mode(
        int(panel.x),
        int(panel.y),
        int(panel.width),
        int(panel.height),
    )
    button: Button
    for button in buttons:
        if is_button_in_clip(button, panel):
            draw_button(button, font)
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
