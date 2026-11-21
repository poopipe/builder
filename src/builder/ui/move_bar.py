"""menu-bar orbit move-speed scale"""

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
from builder.view.orbit_move import clamp_move_speed, nudged_move_speed
from builder.view.viewport import OrbitState

move_panel: str = "move"
move_field_key: str = "speed"
move_label: str = "Move"
move_value_width: float = 56.0


@dataclass(frozen=True)
class MoveBarRects:
    """laid-out move-speed controls"""

    area: Rectangle
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def format_move_speed(value: float) -> str:
    """format move speed for display / editing"""
    return f"{value:.4g}"


def parse_move_speed(text: str) -> float | None:
    """parse a typed move speed"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def layout_move_bar(
    font: Font,
    area: Rectangle,
    min_x: float,
) -> MoveBarRects | None:
    """place move-speed controls after earlier menu widgets; none if no room"""
    height: float = float(ui_button_height)
    label_w: float = (
        measure_text_ex(font, move_label, float(ui_font_size), 0).x + ui_pad
    )
    group_w: float = (
        label_w
        + ui_button_gap
        + float(ui_stepper_width)
        + ui_button_gap
        + move_value_width
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
    value: Rectangle = Rectangle(x, y, move_value_width, height)
    x += move_value_width + ui_button_gap
    plus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    return MoveBarRects(
        area=Rectangle(label.x, y, group_w, height),
        label=label,
        minus=minus,
        value=value,
        plus=plus,
    )


def clear_move_focus(ui: UiState) -> None:
    """drop editing when the move bar is off screen"""
    if ui.focus is not None and ui.focus.panel == move_panel:
        ui.clear_focus()


def focus_move_field(orbit: OrbitState, ui: UiState) -> None:
    """begin typed editing of move speed"""
    ui.focus = FieldId(panel=move_panel, key=move_field_key, owner="")
    ui.edit = begin_edit(format_move_speed(orbit.move_speed))


def commit_move_field(orbit: OrbitState, ui: UiState) -> None:
    """apply the typed move speed; invalid text is discarded"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != move_panel:
        return
    parsed: float | None = parse_move_speed(ui.edit.text)
    if parsed is None:
        return
    orbit.move_speed = clamp_move_speed(parsed)


def handle_move_typing(orbit: OrbitState, ui: UiState) -> None:
    """commit on Enter/Tab, cancel on Escape"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None or ui.focus.panel != move_panel:
        return
    action: TextAction = handle_text_keys(edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    commit_move_field(orbit, ui)
    ui.clear_focus()


def nudge_move_speed(orbit: OrbitState, ui: UiState | None, direction: int) -> None:
    """step move speed up or down"""
    if ui is not None:
        ui.clear_focus()
    orbit.move_speed = nudged_move_speed(orbit.move_speed, direction)


def update_move_bar(
    orbit: OrbitState,
    ui: UiState,
    bar: MoveBarRects,
) -> list[Button]:
    """handle typing and steppers; return buttons to draw"""
    handle_move_typing(orbit, ui)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    if clicked:
        if is_point_in_rect(mouse.x, mouse.y, bar.value):
            focus_move_field(orbit, ui)
        elif is_point_in_rect(mouse.x, mouse.y, bar.area):
            if ui.focus is not None and ui.focus.panel == move_panel:
                commit_move_field(orbit, ui)
                ui.clear_focus()

    buttons: list[Button] = [
        Button(
            label="-",
            on_click=lambda: nudge_move_speed(orbit, ui, -1),
            rect=bar.minus,
        ),
        Button(
            label="+",
            on_click=lambda: nudge_move_speed(orbit, ui, 1),
            rect=bar.plus,
        ),
    ]
    update_buttons(buttons, bar.area)
    return buttons


def draw_move_bar(
    font: Font,
    orbit: OrbitState,
    ui: UiState,
    bar: MoveBarRects,
    buttons: list[Button],
) -> None:
    """draw the move label, editable speed, and stepper buttons"""
    label_pos: Vector2 = Vector2(
        bar.label.x,
        bar.label.y + (bar.label.height - float(ui_font_size)) * 0.5,
    )
    draw_text_ex(font, move_label, label_pos, float(ui_font_size), 0, ui_color_text)
    button: Button
    for button in buttons:
        draw_button(button, font)
    edit: TextEdit | None = ui.focused_edit(move_panel, move_field_key)
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
        format_move_speed(orbit.move_speed),
        focused=False,
        center_unfocused=True,
    )
