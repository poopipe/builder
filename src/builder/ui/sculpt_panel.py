"""sculpt tool panel: brush kind, radius, intensity"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyray import (
    Color,
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    begin_scissor_mode,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    end_scissor_mode,
    get_mouse_position,
    get_mouse_wheel_move,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
)

from builder.heightfield.brushes.brush_types import BrushSpec
from builder.heightfield.brushes.registry import brush_specs
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
    ui_color_border,
    ui_color_button,
    ui_color_button_hover,
    ui_color_button_press,
    ui_color_panel,
    ui_color_selection,
    ui_color_text,
    ui_font_size,
    ui_pad,
    ui_stepper_width,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_buttons

sculpt_panel_id: str = "sculpt"


@dataclass
class SculptBrushRow:
    """one brush kind row"""

    kind: str
    label: str
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False
    is_selected: bool = False


@dataclass
class SculptFloatRow:
    """± / type-in for radius or intensity"""

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def format_sculpt_float(value: float) -> str:
    """format a sculpt float for the type-in field"""
    return f"{float(value):.4g}"


def parse_sculpt_float(text: str) -> float | None:
    """parse typed sculpt float text"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def update_sculpt_panel(
    ui: UiState, area: Rectangle
) -> tuple[list[SculptBrushRow], list[SculptFloatRow], list[Button]]:
    """layout/input for the sculpt panel"""
    buttons: list[Button] = []
    brush_rows: list[SculptBrushRow] = []
    float_rows: list[SculptFloatRow] = []
    title_h: float = float(ui_font_size + ui_pad)
    y: float = area.y + ui_pad + title_h
    x: float = area.x + ui_pad
    inner_w: float = max(0.0, area.width - ui_pad * 2.0)
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    left_down: bool = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)

    specs: tuple[BrushSpec, ...] = brush_specs
    list_h: float = float(len(specs)) * (row_h + ui_button_gap)
    list_rect: Rectangle = Rectangle(area.x, y, area.width, list_h)
    if is_point_in_rect(mouse.x, mouse.y, list_rect):
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            ui.sculpt_panel_scroll -= wheel * (row_h + ui_button_gap)
    ui.sculpt_panel_scroll = max(0.0, ui.sculpt_panel_scroll)

    row_y: float = y - ui.sculpt_panel_scroll
    spec: BrushSpec
    for spec in specs:
        row: SculptBrushRow = SculptBrushRow(
            kind=spec.kind,
            label=spec.label,
            rect=Rectangle(x, row_y, inner_w, row_h),
            is_selected=ui.sculpt_brush_kind == spec.kind,
        )
        row.is_hovered = is_point_in_rect(mouse.x, mouse.y, row.rect)
        row.is_pressed = row.is_hovered and left_down
        if row.is_hovered and left_released:
            ui.sculpt_brush_kind = spec.kind
            ui.clear_focus()
        brush_rows.append(row)
        row_y += row_h + ui_button_gap

    y = list_rect.y + list_rect.height + ui_button_gap

    float_keys: tuple[tuple[str, str], ...] = (
        ("radius", "Radius"),
        ("intensity", "Intensity"),
    )

    if ui.focus is not None and ui.focus.panel == sculpt_panel_id:
        if ui.edit is not None:
            action: TextAction = handle_text_keys(ui.edit)
            match action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    parsed: float | None = parse_sculpt_float(ui.edit.text)
                    if parsed is not None:
                        if ui.focus.key == "radius":
                            ui.sculpt_radius = max(0.5, parsed)
                        elif ui.focus.key == "intensity":
                            ui.sculpt_intensity = parsed
                    if action is TextAction.commit:
                        ui.clear_focus()
                    elif action is TextAction.focus_next:
                        ui.focus = FieldId(
                            panel=sculpt_panel_id,
                            key=(
                                "intensity"
                                if ui.focus.key == "radius"
                                else "radius"
                            ),
                        )
                        value: float = (
                            ui.sculpt_intensity
                            if ui.focus.key == "intensity"
                            else ui.sculpt_radius
                        )
                        ui.edit = begin_edit(format_sculpt_float(value))
                    elif action is TextAction.focus_prev:
                        ui.focus = FieldId(
                            panel=sculpt_panel_id,
                            key=(
                                "radius"
                                if ui.focus.key == "intensity"
                                else "intensity"
                            ),
                        )
                        value = (
                            ui.sculpt_radius
                            if ui.focus.key == "radius"
                            else ui.sculpt_intensity
                        )
                        ui.edit = begin_edit(format_sculpt_float(value))

    key: str
    label: str
    for key, label in float_keys:
        label_rect: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        controls_y: float = y + float(ui_font_size) + 4.0
        minus_rect: Rectangle = Rectangle(x, controls_y, step_w, row_h)
        plus_rect: Rectangle = Rectangle(
            x + inner_w - step_w, controls_y, step_w, row_h
        )
        value_rect: Rectangle = Rectangle(
            minus_rect.x + minus_rect.width + ui_button_gap,
            controls_y,
            plus_rect.x
            - (minus_rect.x + minus_rect.width + ui_button_gap * 2),
            row_h,
        )
        float_rows.append(
            SculptFloatRow(
                key=key,
                label=label_rect,
                minus=minus_rect,
                value=value_rect,
                plus=plus_rect,
            )
        )
        step: float = 1.0 if key == "radius" else 0.05

        def nudge_minus(k: str = key, s: float = step) -> None:
            ui.clear_focus()
            if k == "radius":
                ui.sculpt_radius = max(0.5, ui.sculpt_radius - s)
            else:
                ui.sculpt_intensity = ui.sculpt_intensity - s

        def nudge_plus(k: str = key, s: float = step) -> None:
            ui.clear_focus()
            if k == "radius":
                ui.sculpt_radius = ui.sculpt_radius + s
            else:
                ui.sculpt_intensity = ui.sculpt_intensity + s

        buttons.append(Button(label="-", on_click=nudge_minus, rect=minus_rect))
        buttons.append(Button(label="+", on_click=nudge_plus, rect=plus_rect))
        if clicked and is_point_in_rect(mouse.x, mouse.y, value_rect):
            current: float = (
                ui.sculpt_radius if key == "radius" else ui.sculpt_intensity
            )
            ui.focus = FieldId(panel=sculpt_panel_id, key=key)
            ui.edit = begin_edit(format_sculpt_float(current))
        y = controls_y + row_h + ui_button_gap

    update_buttons(buttons, area)
    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_control: bool = any(
            is_point_in_rect(mouse.x, mouse.y, button.rect) for button in buttons
        )
        on_value: bool = any(
            is_point_in_rect(mouse.x, mouse.y, row.value) for row in float_rows
        )
        on_brush: bool = any(
            is_point_in_rect(mouse.x, mouse.y, row.rect) for row in brush_rows
        )
        if not on_control and not on_value and not on_brush:
            ui.clear_focus()
    return brush_rows, float_rows, buttons


def draw_sculpt_brush_row(font: Font, row: SculptBrushRow) -> None:
    """draw one brush kind row"""
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
    marker: str = "● " if row.is_selected else "  "
    draw_text_ex(
        font,
        marker + row.label,
        Vector2(row.rect.x + 8.0, row.rect.y + 6.0),
        float(ui_font_size),
        0,
        ui_color_text,
    )


def draw_sculpt_panel(
    font: Font,
    area: Rectangle,
    ui: UiState,
    brush_rows: list[SculptBrushRow],
    float_rows: list[SculptFloatRow],
    buttons: list[Button],
) -> None:
    """draw the sculpt panel"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        "Sculpt",
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    title_h: float = float(ui_font_size + ui_pad)
    list_top: float = area.y + ui_pad + title_h
    list_h: float = float(len(brush_rows)) * (
        float(ui_button_height) + ui_button_gap
    )
    begin_scissor_mode(
        int(area.x), int(list_top), int(area.width), int(max(list_h, 1.0))
    )
    row: SculptBrushRow
    for row in brush_rows:
        draw_sculpt_brush_row(font, row)
    end_scissor_mode()

    button: Button
    for button in buttons:
        draw_button(button, font)

    float_row: SculptFloatRow
    for float_row in float_rows:
        label: str = "Radius" if float_row.key == "radius" else "Intensity"
        draw_text_ex(
            font,
            label,
            Vector2(float_row.label.x, float_row.label.y),
            float(ui_font_size),
            0,
            ui_color_text,
        )
        edit: TextEdit | None = ui.focused_edit(sculpt_panel_id, float_row.key)
        value: float = (
            ui.sculpt_radius
            if float_row.key == "radius"
            else ui.sculpt_intensity
        )
        if edit is not None:
            draw_text_field(
                font,
                float_row.value,
                edit.text,
                focused=True,
                caret=edit.caret,
                mark=edit.mark,
            )
        else:
            draw_text_field(
                font,
                float_row.value,
                format_sculpt_float(value),
                focused=False,
                center_unfocused=True,
            )
