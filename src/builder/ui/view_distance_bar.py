"""menu-bar camera far-clip (view distance) scale"""

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
from builder.view.orbit_move import clamp_view_distance, nudged_view_distance
from builder.view.viewport import OrbitState

view_panel: str = "view"
view_field_key: str = "distance"
view_label: str = "View"
view_value_width: float = 72.0


@dataclass(frozen=True)
class ViewDistanceBarRects:
    """laid-out view-distance controls"""

    area: Rectangle
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def format_view_distance(value: float) -> str:
    """format view distance for display / editing"""
    return f"{value:.4g}"


def parse_view_distance(text: str) -> float | None:
    """parse a typed view distance"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def layout_view_distance_bar(
    font: Font,
    area: Rectangle,
    min_x: float,
) -> ViewDistanceBarRects | None:
    """place view-distance controls after earlier menu widgets; none if no room"""
    height: float = float(ui_button_height)
    label_w: float = (
        measure_text_ex(font, view_label, float(ui_font_size), 0).x + ui_pad
    )
    group_w: float = (
        label_w
        + ui_button_gap
        + float(ui_stepper_width)
        + ui_button_gap
        + view_value_width
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
    value: Rectangle = Rectangle(x, y, view_value_width, height)
    x += view_value_width + ui_button_gap
    plus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    return ViewDistanceBarRects(
        area=Rectangle(label.x, y, group_w, height),
        label=label,
        minus=minus,
        value=value,
        plus=plus,
    )


def clear_view_distance_focus(ui: UiState) -> None:
    """drop editing when the view-distance bar is off screen"""
    if ui.focus is not None and ui.focus.panel == view_panel:
        ui.clear_focus()


def focus_view_distance_field(orbit: OrbitState, ui: UiState) -> None:
    """begin typed editing of view distance"""
    ui.focus = FieldId(panel=view_panel, key=view_field_key, owner="")
    ui.edit = begin_edit(format_view_distance(orbit.view_distance))


def commit_view_distance_field(orbit: OrbitState, ui: UiState) -> None:
    """apply the typed view distance; invalid text is discarded"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != view_panel:
        return
    parsed: float | None = parse_view_distance(ui.edit.text)
    if parsed is None:
        return
    orbit.view_distance = clamp_view_distance(parsed)


def handle_view_distance_typing(orbit: OrbitState, ui: UiState) -> None:
    """commit on Enter/Tab, cancel on Escape"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None or ui.focus.panel != view_panel:
        return
    action: TextAction = handle_text_keys(edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    commit_view_distance_field(orbit, ui)
    ui.clear_focus()


def nudge_view_distance_value(
    orbit: OrbitState, ui: UiState | None, direction: int
) -> None:
    """step view distance up or down"""
    if ui is not None:
        ui.clear_focus()
    orbit.view_distance = nudged_view_distance(orbit.view_distance, direction)


def update_view_distance_bar(
    orbit: OrbitState,
    ui: UiState,
    bar: ViewDistanceBarRects,
) -> list[Button]:
    """handle typing and steppers; return buttons to draw"""
    handle_view_distance_typing(orbit, ui)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    if clicked:
        if is_point_in_rect(mouse.x, mouse.y, bar.value):
            focus_view_distance_field(orbit, ui)
        elif is_point_in_rect(mouse.x, mouse.y, bar.area):
            if ui.focus is not None and ui.focus.panel == view_panel:
                commit_view_distance_field(orbit, ui)
                ui.clear_focus()

    buttons: list[Button] = [
        Button(
            label="-",
            on_click=lambda: nudge_view_distance_value(orbit, ui, -1),
            rect=bar.minus,
        ),
        Button(
            label="+",
            on_click=lambda: nudge_view_distance_value(orbit, ui, 1),
            rect=bar.plus,
        ),
    ]
    update_buttons(buttons, bar.area)
    return buttons


def draw_view_distance_bar(
    font: Font,
    orbit: OrbitState,
    ui: UiState,
    bar: ViewDistanceBarRects,
    buttons: list[Button],
) -> None:
    """draw the view label, editable distance, and stepper buttons"""
    label_pos: Vector2 = Vector2(
        bar.label.x,
        bar.label.y + (bar.label.height - float(ui_font_size)) * 0.5,
    )
    draw_text_ex(font, view_label, label_pos, float(ui_font_size), 0, ui_color_text)
    button: Button
    for button in buttons:
        draw_button(button, font)
    edit: TextEdit | None = ui.focused_edit(view_panel, view_field_key)
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
        format_view_distance(orbit.view_distance),
        focused=False,
        center_unfocused=True,
    )
