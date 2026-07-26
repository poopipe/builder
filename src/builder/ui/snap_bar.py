"""menu-bar snap toggle and per-mode snap-step field"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
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
    ui_checkbox_size,
    ui_font_size,
    ui_pad,
    ui_stepper_width,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    Button,
    draw_button,
    draw_checkbox_with_label,
    is_point_in_rect,
    update_buttons,
)
from builder.view.gizmo import GizmoState
from builder.view.gizmo_types import GizmoMode

snap_panel: str = "snap"
snap_field_key: str = "step"
snap_value_width: float = 56.0
snap_toggle_width: float = 72.0

# how much each +/- click nudges the snap increment for the active tool
snap_edit_step: dict[GizmoMode, float] = {
    GizmoMode.translate: 0.25,
    GizmoMode.rotate: 1.0,
    GizmoMode.scale: 0.05,
}
snap_edit_minimum: dict[GizmoMode, float] = {
    GizmoMode.translate: 0.01,
    GizmoMode.rotate: 0.1,
    GizmoMode.scale: 0.01,
}


@dataclass(frozen=True)
class SnapBarRects:
    """laid-out snap controls"""

    area: Rectangle
    toggle: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def format_snap_value(value: float) -> str:
    """format the snap increment for display / editing"""
    return f"{value:.4g}"


def parse_snap_value(text: str) -> float | None:
    """parse a typed snap value"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def layout_snap_bar(
    font: Font,
    area: Rectangle,
    min_x: float,
) -> SnapBarRects | None:
    """place snap controls after the menu buttons; none if they will not fit"""
    height: float = float(ui_button_height)
    toggle_w: float = max(
        snap_toggle_width,
        measure_text_ex(font, "Snap", float(ui_font_size), 0).x
        + float(ui_checkbox_size)
        + ui_pad * 2.0,
    )
    group_w: float = (
        toggle_w
        + ui_button_gap
        + float(ui_stepper_width)
        + ui_button_gap
        + snap_value_width
        + ui_button_gap
        + float(ui_stepper_width)
    )
    x: float = min_x
    if x + group_w + ui_pad > area.x + area.width:
        return None
    y: float = area.y + (area.height - height) * 0.5
    toggle: Rectangle = Rectangle(x, y, toggle_w, height)
    x += toggle_w + ui_button_gap
    minus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    x += float(ui_stepper_width) + ui_button_gap
    value: Rectangle = Rectangle(x, y, snap_value_width, height)
    x += snap_value_width + ui_button_gap
    plus: Rectangle = Rectangle(x, y, float(ui_stepper_width), height)
    return SnapBarRects(
        area=Rectangle(toggle.x, y, group_w, height),
        toggle=toggle,
        minus=minus,
        value=value,
        plus=plus,
    )


def clear_snap_focus(ui: UiState) -> None:
    """drop editing when the snap bar is off screen"""
    if ui.focus is not None and ui.focus.panel == snap_panel:
        ui.clear_focus()


def sync_snap_focus(ui: UiState, mode: GizmoMode) -> None:
    """drop editing when the gizmo tool changes under it"""
    if ui.focus is None or ui.focus.panel != snap_panel:
        return
    if ui.focus.owner != mode.value:
        ui.clear_focus()


def focus_snap_field(state: GizmoState, ui: UiState) -> None:
    """begin typed editing of the active tool's snap increment"""
    value: float = state.snap_step[state.mode]
    ui.focus = FieldId(panel=snap_panel, key=snap_field_key, owner=state.mode.value)
    ui.edit = begin_edit(format_snap_value(value))


def commit_snap_field(state: GizmoState, ui: UiState) -> None:
    """apply the typed snap value; invalid text is discarded"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != snap_panel:
        return
    parsed: float | None = parse_snap_value(ui.edit.text)
    if parsed is None:
        return
    mode: GizmoMode = state.mode
    state.snap_step[mode] = max(snap_edit_minimum[mode], parsed)


def handle_snap_typing(state: GizmoState, ui: UiState) -> None:
    """commit on Enter/Tab, cancel on Escape"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None or ui.focus.panel != snap_panel:
        return
    action: TextAction = handle_text_keys(edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    commit_snap_field(state, ui)
    ui.clear_focus()


def nudge_snap_step(state: GizmoState, ui: UiState, direction: int) -> None:
    """step the active tool's snap increment up or down"""
    ui.clear_focus()
    mode: GizmoMode = state.mode
    step: float = snap_edit_step[mode]
    minimum: float = snap_edit_minimum[mode]
    next_value: float = state.snap_step[mode] + step * float(direction)
    # keep float noise from accumulating on repeated clicks
    next_value = round(next_value / step) * step
    state.snap_step[mode] = max(minimum, next_value)


def update_snap_bar(
    state: GizmoState,
    ui: UiState,
    bar: SnapBarRects,
) -> list[Button]:
    """handle toggle, typing, and steppers; return buttons to draw"""
    sync_snap_focus(ui, state.mode)
    handle_snap_typing(state, ui)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    if clicked:
        if is_point_in_rect(mouse.x, mouse.y, bar.toggle):
            ui.clear_focus()
            state.snap_enabled = not state.snap_enabled
        elif is_point_in_rect(mouse.x, mouse.y, bar.value):
            focus_snap_field(state, ui)
        elif is_point_in_rect(mouse.x, mouse.y, bar.area):
            if ui.focus is not None and ui.focus.panel == snap_panel:
                commit_snap_field(state, ui)
                ui.clear_focus()

    buttons: list[Button] = [
        Button(
            label="-",
            on_click=lambda: nudge_snap_step(state, ui, -1),
            rect=bar.minus,
        ),
        Button(
            label="+",
            on_click=lambda: nudge_snap_step(state, ui, 1),
            rect=bar.plus,
        ),
    ]
    update_buttons(buttons, bar.area)
    return buttons


def draw_snap_bar(
    font: Font,
    state: GizmoState,
    ui: UiState,
    bar: SnapBarRects,
    buttons: list[Button],
) -> None:
    """draw the snap checkbox, editable value, and stepper buttons"""
    draw_checkbox_with_label(
        font,
        bar.toggle,
        "Snap",
        checked=state.snap_enabled,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)
    edit: TextEdit | None = ui.focused_edit(
        snap_panel, snap_field_key, state.mode.value
    )
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
        format_snap_value(state.snap_step[state.mode]),
        focused=False,
        center_unfocused=True,
    )
