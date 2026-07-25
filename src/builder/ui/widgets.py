"""UI widgets: layout, update, and draw as separate steps"""

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
    draw_circle,
    draw_circle_lines,
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
    ui_checkbox_size,
    ui_color_border,
    ui_color_button,
    ui_color_button_hover,
    ui_color_button_press,
    ui_color_checkbox_on,
    ui_color_input,
    ui_color_menu,
    ui_color_panel,
    ui_color_status,
    ui_color_text,
    ui_font_size,
    ui_menu_bar_height,
    ui_pad,
    ui_side_panel_width,
    ui_status_bar_height,
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
    """window regions for the current frame"""

    menu: Rectangle
    panel: Rectangle
    outliner: Rectangle
    viewport: Rectangle
    meshes: Rectangle
    inspector: Rectangle
    status: Rectangle


def compute_layout(
    width: int,
    height: int,
    panel_width: int = ui_side_panel_width,
    outliner_width: int = 0,
    meshes_width: int = 0,
    inspector_width: int = 0,
) -> LayoutRects:
    """split the window into menu, side panels, viewport, and status bar

    columns left to right: tools panel, outliner, viewport, inspector, meshes
    """
    menu_h: int = ui_menu_bar_height
    status_h: int = ui_status_bar_height
    panel_w: int = max(0, panel_width)
    outliner_w: int = max(0, outliner_width)
    meshes_w: int = max(0, meshes_width)
    inspector_w: int = max(0, inspector_width)
    body_y: int = menu_h
    body_h: int = max(0, height - menu_h - status_h)
    viewport_w: int = max(
        0, width - panel_w - outliner_w - meshes_w - inspector_w
    )
    outliner_x: float = float(panel_w)
    viewport_x: float = float(panel_w + outliner_w)
    inspector_x: float = viewport_x + float(viewport_w)
    meshes_x: float = inspector_x + float(inspector_w)
    return LayoutRects(
        menu=Rectangle(0, 0, float(width), float(menu_h)),
        panel=Rectangle(0, float(body_y), float(panel_w), float(body_h)),
        outliner=Rectangle(
            outliner_x,
            float(body_y),
            float(outliner_w),
            float(body_h),
        ),
        viewport=Rectangle(
            viewport_x,
            float(body_y),
            float(viewport_w),
            float(body_h),
        ),
        inspector=Rectangle(
            inspector_x,
            float(body_y),
            float(inspector_w),
            float(body_h),
        ),
        meshes=Rectangle(meshes_x, float(body_y), float(meshes_w), float(body_h)),
        status=Rectangle(
            0,
            float(height - status_h),
            float(width),
            float(status_h),
        ),
    )


def is_point_in_rect(x: float, y: float, rect: Rectangle) -> bool:
    """return True if (x, y) lies inside rect"""
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def is_button_in_clip(button: Button, clip: Rectangle) -> bool:
    """return True if the button intersects the clip rectangle"""
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
    """refresh hover/press state and fire ``on_click`` on release"""
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
    """draw a button from its current interaction state"""
    color: Color
    if button.is_pressed:
        color = ui_color_button_press
    elif button.is_hovered:
        color = ui_color_button_hover
    else:
        color = ui_color_button

    draw_rectangle_rec(button.rect, color)
    draw_rectangle_lines_ex(button.rect, 1, ui_color_border)

    text_size: int = ui_font_size
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
        ui_color_text,
    )


def checkbox_box_rect(hit: Rectangle) -> Rectangle:
    """place the indicator square on the left of a checkbox hit area"""
    size: float = float(ui_checkbox_size)
    return Rectangle(
        hit.x,
        hit.y + (hit.height - size) * 0.5,
        size,
        size,
    )


def draw_checkbox_with_label(
    font: Font,
    hit: Rectangle,
    label: str,
    *,
    checked: bool,
) -> None:
    """draw a labeled checkbox: filled square when on, empty when off"""
    box: Rectangle = checkbox_box_rect(hit)
    draw_rectangle_rec(box, ui_color_checkbox_on if checked else ui_color_input)
    draw_rectangle_lines_ex(box, 1.0, ui_color_border)
    size: float = float(ui_font_size)
    draw_text_ex(
        font,
        label,
        Vector2(
            box.x + box.width + ui_pad,
            hit.y + (hit.height - size) * 0.5,
        ),
        size,
        0,
        ui_color_text,
    )


def draw_radio_option(
    font: Font,
    hit: Rectangle,
    label: str,
    *,
    selected: bool,
) -> None:
    """draw one labeled radio option: filled dot when selected, ring when not"""
    radius: float = float(ui_checkbox_size) * 0.5
    cx: int = int(hit.x + radius)
    cy: int = int(hit.y + hit.height * 0.5)
    draw_circle_lines(cx, cy, radius, ui_color_border)
    if selected:
        draw_circle(cx, cy, radius - 3.0, ui_color_checkbox_on)
    size: float = float(ui_font_size)
    draw_text_ex(
        font,
        label,
        Vector2(
            hit.x + radius * 2.0 + 6.0,
            hit.y + (hit.height - size) * 0.5,
        ),
        size,
        0,
        ui_color_text,
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
    """place buttons left to right inside area"""
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
    """place buttons top to bottom inside area"""
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
    """draw the menu bar background and its buttons"""
    draw_rectangle_rec(bar, ui_color_menu)
    draw_line(
        int(bar.x),
        int(bar.y + bar.height - 1),
        int(bar.x + bar.width),
        int(bar.y + bar.height - 1),
        ui_color_border,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)


def draw_button_stack(font: Font, panel: Rectangle, buttons: Sequence[Button]) -> None:
    """draw the side panel background and its clipped buttons"""
    draw_rectangle_rec(panel, ui_color_panel)
    draw_line(
        int(panel.x + panel.width - 1),
        int(panel.y),
        int(panel.x + panel.width - 1),
        int(panel.y + panel.height),
        ui_color_border,
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
    """draw a single-line status bar"""
    draw_rectangle_rec(rect, ui_color_status)
    draw_text_ex(
        font,
        text,
        Vector2(rect.x + ui_pad, rect.y + 4),
        float(ui_font_size - 2),
        0,
        ui_color_text,
    )
