"""modal single-select list picker dialog"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pyray import (
    Color,
    Font,
    KeyboardKey,
    MouseButton,
    Rectangle,
    Vector2,
    begin_scissor_mode,
    draw_rectangle,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    end_scissor_mode,
    get_mouse_position,
    get_mouse_wheel_move,
    get_time,
    is_key_pressed,
    is_mouse_button_down,
    is_mouse_button_released,
)

from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_button,
    ui_color_button_hover,
    ui_color_button_press,
    ui_color_overlay,
    ui_color_panel,
    ui_color_selection,
    ui_color_text,
    ui_double_click_sec,
    ui_file_browser_height,
    ui_file_browser_width,
    ui_font_size,
    ui_pad,
)
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_button


class ListPickerPurpose(IntEnum):
    road_kit = 0
    road_move_usage = 1
    vector_brush = 2


# sentinel list item: create a custom-named usage group
list_picker_new_item: str = "New…"


class ListPickerCancel(IntEnum):
    cancelled = 0


type ListPickerFrameResult = str | ListPickerCancel | None


@dataclass
class ListPickerState:
    """mutable modal list picker session"""

    purpose: ListPickerPurpose
    title: str
    items: tuple[str, ...]
    selected: int = 0
    target_id: str | None = None
    scroll_y: float = 0.0
    last_click_index: int = -1
    last_click_time: float = 0.0


@dataclass
class ListPickerRow:
    """laid-out list row"""

    index: int
    label: str
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False
    is_selected: bool = False


def make_list_picker(
    *,
    purpose: ListPickerPurpose,
    title: str,
    items: tuple[str, ...] | list[str],
    selected: int = 0,
    target_id: str | None = None,
) -> ListPickerState:
    """open a list picker; selected is clamped into range"""
    item_tuple: tuple[str, ...] = tuple(items)
    index: int = 0
    if item_tuple:
        index = max(0, min(int(selected), len(item_tuple) - 1))
    return ListPickerState(
        purpose=purpose,
        title=title,
        items=item_tuple,
        selected=index,
        target_id=target_id,
    )


def list_picker_window_rect(screen_w: int, screen_h: int) -> Rectangle:
    """centered dialog rect"""
    width: float = min(ui_file_browser_width * 0.7, float(screen_w) - 40.0)
    height: float = min(ui_file_browser_height * 0.85, float(screen_h) - 40.0)
    return Rectangle(
        (float(screen_w) - width) * 0.5,
        (float(screen_h) - height) * 0.5,
        width,
        height,
    )


def update_list_picker(
    state: ListPickerState,
    screen_w: int,
    screen_h: int,
) -> tuple[ListPickerFrameResult, list[ListPickerRow], list[Button], Rectangle]:
    """handle input; return (result, rows, buttons, window)"""
    window: Rectangle = list_picker_window_rect(screen_w, screen_h)
    title_h: float = float(ui_font_size + ui_pad)
    footer_h: float = float(ui_button_height + ui_pad * 2)
    list_top: float = window.y + ui_pad + title_h + ui_pad
    list_h: float = max(
        0.0,
        window.height - ui_pad * 2.0 - title_h - ui_pad - footer_h,
    )
    list_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        list_top,
        window.width - ui_pad * 2.0,
        list_h,
    )
    btn_y: float = window.y + window.height - ui_pad - float(ui_button_height)
    btn_w: float = 88.0
    cancel_rect: Rectangle = Rectangle(
        window.x + window.width - ui_pad - btn_w,
        btn_y,
        btn_w,
        float(ui_button_height),
    )
    confirm_rect: Rectangle = Rectangle(
        cancel_rect.x - ui_button_gap - btn_w,
        btn_y,
        btn_w,
        float(ui_button_height),
    )

    if is_key_pressed(KeyboardKey.KEY_ESCAPE):
        return ListPickerCancel.cancelled, [], [], window

    result: ListPickerFrameResult = None
    row_h: float = float(ui_button_height)
    content_h: float = max(
        0.0, len(state.items) * (row_h + ui_button_gap) - ui_button_gap
    )
    max_scroll: float = max(0.0, content_h - list_h)
    mouse: Vector2 = get_mouse_position()
    if is_point_in_rect(mouse.x, mouse.y, list_rect):
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            state.scroll_y -= wheel * row_h
    state.scroll_y = max(0.0, min(max_scroll, state.scroll_y))

    left_down: bool = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    pointer_in_list: bool = is_point_in_rect(mouse.x, mouse.y, list_rect)

    rows: list[ListPickerRow] = []
    y: float = list_top - state.scroll_y
    index: int
    label: str
    for index, label in enumerate(state.items):
        row: ListPickerRow = ListPickerRow(
            index=index,
            label=label,
            rect=Rectangle(list_rect.x, y, list_rect.width, row_h),
            is_selected=state.selected == index,
        )
        row.is_hovered = is_point_in_rect(mouse.x, mouse.y, row.rect)
        row.is_pressed = row.is_hovered and left_down
        visible: bool = (
            row.rect.y + row.rect.height > list_rect.y
            and row.rect.y < list_rect.y + list_rect.height
        )
        if pointer_in_list and visible and row.is_hovered and left_released:
            state.selected = index
            now: float = get_time()
            is_double: bool = (
                state.last_click_index == index
                and (now - state.last_click_time) <= ui_double_click_sec
            )
            state.last_click_index = index
            state.last_click_time = now
            if is_double:
                result = label
        rows.append(row)
        y += row_h + ui_button_gap

    def confirm() -> None:
        nonlocal result
        if not state.items:
            return
        pick: int = max(0, min(state.selected, len(state.items) - 1))
        result = state.items[pick]

    def cancel() -> None:
        nonlocal result
        result = ListPickerCancel.cancelled

    buttons: list[Button] = [
        Button("OK", confirm, confirm_rect),
        Button("Cancel", cancel, cancel_rect),
    ]
    button: Button
    for button in buttons:
        update_button(button, mouse, left_down, left_released)

    return result, rows, buttons, window


def draw_list_picker_row(font: Font, row: ListPickerRow) -> None:
    """draw one picker row"""
    color: Color
    if row.is_selected:
        color = ui_color_selection
    elif row.is_pressed:
        color = ui_color_button_press
    elif row.is_hovered:
        color = ui_color_button_hover
    else:
        color = ui_color_button
    draw_rectangle_rec(row.rect, color)
    draw_rectangle_lines_ex(row.rect, 1, ui_color_border)
    draw_text_ex(
        font,
        row.label,
        Vector2(row.rect.x + 8.0, row.rect.y + 4.0),
        float(ui_font_size),
        0,
        ui_color_text,
    )


def draw_list_picker(
    font: Font,
    state: ListPickerState,
    rows: list[ListPickerRow],
    buttons: list[Button],
    window: Rectangle,
    screen_w: int,
    screen_h: int,
) -> None:
    """draw dim overlay and list picker dialog"""
    draw_rectangle(0, 0, screen_w, screen_h, ui_color_overlay)
    draw_rectangle_rec(window, ui_color_panel)
    draw_rectangle_lines_ex(window, 1, ui_color_border)

    draw_text_ex(
        font,
        state.title,
        Vector2(window.x + ui_pad, window.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    title_h: float = float(ui_font_size + ui_pad)
    footer_h: float = float(ui_button_height + ui_pad * 2)
    list_top: float = window.y + ui_pad + title_h + ui_pad
    list_h: float = max(
        0.0,
        window.height - ui_pad * 2.0 - title_h - ui_pad - footer_h,
    )
    list_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        list_top,
        window.width - ui_pad * 2.0,
        list_h,
    )
    begin_scissor_mode(
        int(list_rect.x),
        int(list_rect.y),
        int(list_rect.width),
        int(list_rect.height),
    )
    row: ListPickerRow
    for row in rows:
        draw_list_picker_row(font, row)
    end_scissor_mode()

    button: Button
    for button in buttons:
        draw_button(button, font)
