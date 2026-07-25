"""Right-side inspector for parametric group recipes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyray import (
    Font,
    KeyboardKey,
    MouseButton,
    Rectangle,
    Vector2,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    get_char_pressed,
    get_mouse_position,
    get_time,
    is_key_pressed,
    is_mouse_button_pressed,
    measure_text_ex,
    set_exit_key,
)

from builder.generators.generator_types import Generator, ParamField, ParamValue
from builder.generators.registry import (
    field_for,
    format_param,
    get_spec,
    parse_param,
)
from builder.generators.regenerate import (
    bake_group,
    step_generator_param,
    set_generator_param,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.theme import (
    BUTTON_GAP,
    BUTTON_HEIGHT,
    COLOUR_BORDER,
    COLOUR_BUTTON,
    COLOUR_INPUT,
    COLOUR_INPUT_FOCUS,
    COLOUR_PANEL,
    COLOUR_SELECTION,
    COLOUR_TEXT,
    FONT_SIZE,
    PAD,
    STEPPER_WIDTH,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    Button,
    draw_button,
    is_point_in_rect,
    update_buttons,
)


@dataclass(frozen=True)
class ParamRowRects:
    """ hit targets for one inspector param row """

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def sync_inspector_focus(ui: UiState, group: Node | None) -> None:
    """ clear draft editing when the inspected group changes """
    if group is None:
        ui.clear_inspector_focus()
        return
    if ui.inspector_group_id is not None and ui.inspector_group_id != group.id:
        ui.clear_inspector_focus()


def apply_inspector_exit_key(ui: UiState) -> None:
    """ disable window-close-on-escape while typing in the inspector """
    if ui.inspector_focus_key is not None:
        set_exit_key(0)
    else:
        set_exit_key(KeyboardKey.KEY_ESCAPE)


def layout_param_rows(area: Rectangle, fields: tuple[ParamField, ...]) -> list[ParamRowRects]:
    """ place label / stepper / value rows under the inspector title """
    rows: list[ParamRowRects] = []
    y: float = area.y + PAD + float(FONT_SIZE) + PAD
    x: float = area.x + PAD
    inner_w: float = area.width - PAD * 2
    field: ParamField
    for field in fields:
        label: Rectangle = Rectangle(x, y, inner_w, float(FONT_SIZE))
        y += float(FONT_SIZE) + 4.0
        controls_y: float = y
        minus: Rectangle = Rectangle(x, controls_y, float(STEPPER_WIDTH), float(BUTTON_HEIGHT))
        plus: Rectangle = Rectangle(
            x + inner_w - float(STEPPER_WIDTH),
            controls_y,
            float(STEPPER_WIDTH),
            float(BUTTON_HEIGHT),
        )
        value: Rectangle = Rectangle(
            minus.x + minus.width + BUTTON_GAP,
            controls_y,
            plus.x - (minus.x + minus.width + BUTTON_GAP * 2),
            float(BUTTON_HEIGHT),
        )
        rows.append(
            ParamRowRects(
                key=field.key,
                label=label,
                minus=minus,
                value=value,
                plus=plus,
            )
        )
        y += float(BUTTON_HEIGHT) + PAD
    return rows


def bake_button_rect(area: Rectangle, rows: list[ParamRowRects]) -> Rectangle:
    """ place the Bake button below the last param row """
    y: float = area.y + PAD + float(FONT_SIZE) + PAD
    if rows:
        last: ParamRowRects = rows[-1]
        y = last.plus.y + last.plus.height + PAD
    return Rectangle(
        area.x + PAD,
        y,
        area.width - PAD * 2,
        float(BUTTON_HEIGHT),
    )


def focus_param_field(ui: UiState, group: Node, field: ParamField) -> None:
    """ begin typed editing for one param """
    generator: Generator | None = group.generator
    if generator is None:
        return
    value: ParamValue = generator.params[field.key]
    ui.inspector_group_id = group.id
    ui.inspector_focus_key = field.key
    ui.inspector_draft = format_param(field, value)
    ui.inspector_select_all = True


def commit_inspector_draft(scene: Scene, ui: UiState, group: Node) -> None:
    """ apply typed draft on Enter; invalid text is discarded """
    if ui.inspector_focus_key is None or ui.inspector_draft is None:
        ui.clear_inspector_focus()
        return
    generator: Generator | None = group.generator
    if generator is None:
        ui.clear_inspector_focus()
        return
    field: ParamField = field_for(get_spec(generator.kind), ui.inspector_focus_key)
    parsed: ParamValue | None = parse_param(field, ui.inspector_draft)
    ui.clear_inspector_focus()
    if parsed is None:
        return
    set_generator_param(scene, group.id, field.key, parsed)


def handle_inspector_typing(scene: Scene, ui: UiState, group: Node) -> None:
    """ typed-field keyboard: commit on Enter, cancel on Escape """
    if ui.inspector_focus_key is None or ui.inspector_draft is None:
        return
    if is_key_pressed(KeyboardKey.KEY_ESCAPE):
        ui.clear_inspector_focus()
        return
    if is_key_pressed(KeyboardKey.KEY_ENTER) or is_key_pressed(
        KeyboardKey.KEY_KP_ENTER
    ):
        commit_inspector_draft(scene, ui, group)
        return
    if is_key_pressed(KeyboardKey.KEY_BACKSPACE):
        if ui.inspector_select_all:
            ui.inspector_draft = ""
            ui.inspector_select_all = False
        else:
            ui.inspector_draft = ui.inspector_draft[:-1]
        return
    code: int = get_char_pressed()
    while code > 0:
        char: str = chr(code)
        if char.isprintable():
            if ui.inspector_select_all:
                ui.inspector_draft = char
                ui.inspector_select_all = False
            else:
                ui.inspector_draft = (ui.inspector_draft or "") + char
        code = get_char_pressed()


def make_stepper_button(
    label: str,
    rect: Rectangle,
    on_click: Callable[[], None],
) -> Button:
    """ build a +/- stepper button at a fixed rect """
    return Button(label=label, on_click=on_click, rect=rect)


def update_inspector(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    group: Node,
) -> tuple[list[Button], list[ParamRowRects]]:
    """ handle inspector input; return stepper/bake buttons and row geometry """
    generator: Generator | None = group.generator
    if generator is None:
        return [], []
    spec = get_spec(generator.kind)
    rows: list[ParamRowRects] = layout_param_rows(area, spec.fields)
    handle_inspector_typing(scene, ui, group)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    buttons: list[Button] = []
    row: ParamRowRects
    for row in rows:
        field: ParamField = field_for(spec, row.key)
        key: str = row.key

        def step_minus(k: str = key) -> None:
            ui.clear_inspector_focus()
            step_generator_param(scene, group.id, k, -1)

        def step_plus(k: str = key) -> None:
            ui.clear_inspector_focus()
            step_generator_param(scene, group.id, k, 1)

        buttons.append(make_stepper_button("-", row.minus, step_minus))
        buttons.append(make_stepper_button("+", row.plus, step_plus))
        if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
            focus_param_field(ui, group, field)

    bake_rect: Rectangle = bake_button_rect(area, rows)

    def on_bake() -> None:
        ui.clear_inspector_focus()
        bake_group(scene, group.id)
        ui.status = "Baked group (static)"

    buttons.append(Button(label="Bake", on_click=on_bake, rect=bake_rect))
    update_buttons(buttons, area)

    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_control: bool = False
        button: Button
        for button in buttons:
            if is_point_in_rect(mouse.x, mouse.y, button.rect):
                on_control = True
                break
        on_value: bool = False
        for row in rows:
            if is_point_in_rect(mouse.x, mouse.y, row.value):
                on_value = True
                break
        if not on_control and not on_value:
            ui.clear_inspector_focus()

    return buttons, rows


def draw_value_field(
    font: Font,
    rect: Rectangle,
    text: str,
    *,
    focused: bool,
    select_all: bool,
) -> None:
    """ draw the editable value box for one param """
    draw_rectangle_rec(rect, COLOUR_INPUT if focused else COLOUR_BUTTON)
    border = COLOUR_INPUT_FOCUS if focused else COLOUR_BORDER
    draw_rectangle_lines_ex(rect, 2.0 if focused else 1.0, border)

    pad_x: float = 6.0
    text_w: float = measure_text_ex(font, text, float(FONT_SIZE), 0).x
    ty: float = rect.y + (rect.height - float(FONT_SIZE)) * 0.5
    tx: float
    if focused:
        tx = rect.x + pad_x
    else:
        tx = rect.x + (rect.width - text_w) * 0.5

    if focused and select_all and text != "":
        draw_rectangle_rec(
            Rectangle(tx - 1.0, ty - 1.0, text_w + 2.0, float(FONT_SIZE) + 2.0),
            COLOUR_SELECTION,
        )

    draw_text_ex(font, text, Vector2(tx, ty), float(FONT_SIZE), 0, COLOUR_TEXT)

    if focused and not select_all and int(get_time() * 2.0) % 2 == 0:
        caret_x: float = tx + text_w + 1.0
        draw_rectangle_rec(
            Rectangle(caret_x, ty, 1.0, float(FONT_SIZE)),
            COLOUR_TEXT,
        )


def draw_inspector(
    font: Font,
    area: Rectangle,
    ui: UiState,
    group: Node,
    buttons: list[Button],
    rows: list[ParamRowRects],
) -> None:
    """ draw the right inspector panel for a parametric group """
    generator: Generator | None = group.generator
    if generator is None:
        return
    spec = get_spec(generator.kind)
    draw_rectangle_rec(area, COLOUR_PANEL)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        COLOUR_BORDER,
    )
    draw_text_ex(
        font,
        spec.label,
        Vector2(area.x + PAD, area.y + PAD),
        float(FONT_SIZE),
        0,
        COLOUR_TEXT,
    )
    row: ParamRowRects
    for row in rows:
        field: ParamField = field_for(spec, row.key)
        draw_text_ex(
            font,
            field.label,
            Vector2(row.label.x, row.label.y),
            float(FONT_SIZE),
            0,
            COLOUR_TEXT,
        )
        display: str
        focused: bool
        select_all: bool = False
        if (
            ui.inspector_focus_key == row.key
            and ui.inspector_group_id == group.id
            and ui.inspector_draft is not None
        ):
            display = ui.inspector_draft
            focused = True
            select_all = ui.inspector_select_all
        else:
            display = format_param(field, generator.params[row.key])
            focused = False
        draw_value_field(
            font,
            row.value,
            display,
            focused=focused,
            select_all=select_all,
        )

    button: Button
    for button in buttons:
        draw_button(button, font)
