"""menu-bar orbit zoom-speed scale"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_pressed,
    measure_text_ex,
)

from builder.ui.text_field import (
    FieldId,
    TextAction,
    TextEdit,
    begin_edit,
    draw_text_field,
    handle_text_keys,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_text,
    ui_font_size,
    ui_pad,
    ui_stepper_width,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    Button,
    draw_button,
    is_point_in_rect,
    update_buttons,
)
from builder.view.viewport import OrbitState

zoom_panel: str = "zoom"
zoom_field_key: str = "speed"
zoom_label: str = "Zoom"
zoom_value_width: float = 56.0
zoom_edit_step: float = 0.25
zoom_edit_minimum: float = 0.1
zoom_edit_maximum: float = 100.0


@dataclass(frozen=True)
class ZoomBarRects:
    """laid-out zoom-speed controls"""

    area: Rectangle
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def format_zoom_speed(value: float) -> str:
    """format zoom speed for display / editing"""
    return f"{value:.4g}"


def parse_zoom_speed(text: str) -> float | None:
    """parse a typed zoom speed"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def clamp_zoom_speed(value: float) -> float:
    """keep zoom speed in a usable range"""
    return max(zoom_edit_minimum, min(zoom_edit_maximum, value))


def layout_zoom_bar(
    font: Font,
    area: Rectangle,
    min_x: float,
) -> ZoomBarRects | None:
    """place zoom-speed controls after earlier menu widgets; none if no room"""
    height: float = float(ui_button_height)
    label_w: float = (
        measure_text_ex(font, zoom_label, float(ui_font_size), 0).x + ui_pad
    )
    group_w: float = (
        label_w
        + ui_button_gap
        + float(ui_stepper_width)
        + ui_button_gap
        + zoom_value_width
        + ui_button_gap
        + float(ui_stepper_width)
    )
    x: float = min_x
    if x + group_w + ui_pad > area.x + area.width:
        return None
    y: float = area.y + (area.height - height) * 0.5
    label: Rectangle = Rectangle(x, y, label_w, height)
    x += label_w + ui_button_gap
    minus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    x += float(ui_stepper_width) + ui_button_gap
    value: Rectangle = Rectangle(x, y, zoom_value_width, height)
    x += zoom_value_width + ui_button_gap
    plus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    return ZoomBarRects(
        area=Rectangle(label.x, y, group_w, height),
        label=label,
        minus=minus,
        value=value,
        plus=plus,
    )


def clear_zoom_focus(ui: UiState) -> None:
    """drop editing when the zoom bar is off screen"""
    if ui.focus is not None and ui.focus.panel == zoom_panel:
        ui.clear_focus()


def focus_zoom_field(orbit: OrbitState, ui: UiState) -> None:
    """begin typed editing of zoom speed"""
    ui.focus = FieldId(panel=zoom_panel, key=zoom_field_key, owner="")
    ui.edit = begin_edit(format_zoom_speed(orbit.zoom_speed))


def commit_zoom_field(orbit: OrbitState, ui: UiState) -> None:
    """apply the typed zoom speed; invalid text is discarded"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != zoom_panel:
        return
    parsed: float | None = parse_zoom_speed(ui.edit.text)
    if parsed is None:
        return
    orbit.zoom_speed = clamp_zoom_speed(parsed)


def handle_zoom_typing(orbit: OrbitState, ui: UiState) -> None:
    """commit on Enter/Tab, cancel on Escape"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None or ui.focus.panel != zoom_panel:
        return
    action: TextAction = handle_text_keys(edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    commit_zoom_field(orbit, ui)
    ui.clear_focus()


def nudge_zoom_speed(orbit: OrbitState, ui: UiState, direction: int) -> None:
    """step zoom speed up or down"""
    ui.clear_focus()
    next_value: float = orbit.zoom_speed + zoom_edit_step * float(direction)
    next_value = round(next_value / zoom_edit_step) * zoom_edit_step
    orbit.zoom_speed = clamp_zoom_speed(next_value)


def update_zoom_bar(
    orbit: OrbitState,
    ui: UiState,
    bar: ZoomBarRects,
) -> list[Button]:
    """handle typing and steppers; return buttons to draw"""
    handle_zoom_typing(orbit, ui)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    if clicked:
        if is_point_in_rect(mouse.x, mouse.y, bar.value):
            focus_zoom_field(orbit, ui)
        elif is_point_in_rect(mouse.x, mouse.y, bar.area):
            if ui.focus is not None and ui.focus.panel == zoom_panel:
                commit_zoom_field(orbit, ui)
                ui.clear_focus()

    buttons: list[Button] = [
        Button(
            label="-",
            on_click=lambda: nudge_zoom_speed(orbit, ui, -1),
            rect=bar.minus,
        ),
        Button(
            label="+",
            on_click=lambda: nudge_zoom_speed(orbit, ui, 1),
            rect=bar.plus,
        ),
    ]
    update_buttons(buttons, bar.area)
    return buttons


def draw_zoom_bar(
    font: Font,
    orbit: OrbitState,
    ui: UiState,
    bar: ZoomBarRects,
    buttons: list[Button],
) -> None:
    """draw the zoom label, editable speed, and stepper buttons"""
    label_pos: Vector2 = Vector2(
        bar.label.x,
        bar.label.y + (bar.label.height - float(ui_font_size)) * 0.5,
    )
    draw_text_ex(font, zoom_label, label_pos, float(ui_font_size), 0, ui_color_text)
    button: Button
    for button in buttons:
        draw_button(button, font)
    edit: TextEdit | None = ui.focused_edit(zoom_panel, zoom_field_key)
    if edit is not None:
        draw_text_field(
            font,
            bar.value,
            edit.text,
            focused=True,
            caret=edit.caret,
            mark=edit.mark,
        )
        return
    draw_text_field(
        font,
        bar.value,
        format_zoom_speed(orbit.zoom_speed),
        focused=False,
        center_unfocused=True,
    )
