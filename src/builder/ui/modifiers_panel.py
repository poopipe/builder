"""modifiers column: stack editor for the selected generator group"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_pressed,
)

from builder.generators.generator_types import (
    Generator,
    Modifier,
    ParamField,
    ParamValue,
)
from builder.generators.registry import format_param, parse_param
from builder.generators.regenerate import (
    set_generator_modifiers,
    set_modifier_param,
    step_modifier_param,
)
from builder.modifiers.modifier_types import ModifierSpec
from builder.modifiers.registry import (
    get_modifier_spec,
    known_modifier_kind,
    make_modifier,
    modifier_field_for,
    modifier_params_with_defaults,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.inspector import ParamRowRects, layout_field_row
from builder.ui.text_field import (
    FieldId,
    TextAction,
    TextEdit,
    begin_edit,
    draw_text_field,
    field_after,
    handle_text_keys,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_panel,
    ui_color_text,
    ui_font_size,
    ui_pad,
    ui_stepper_width,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import (
    Button,
    draw_button,
    draw_checkbox_with_label,
    draw_radio_option,
    is_point_in_rect,
    update_buttons,
)


modifiers_panel: str = "modifiers"


def sync_modifiers_focus(ui: UiState, group: Node | None) -> None:
    """clear draft editing when the selected generator changes"""
    if ui.focus is None or ui.focus.panel != modifiers_panel:
        return
    if group is None or ui.focus.owner != group.id:
        ui.clear_focus()


def parse_mod_focus_key(key: str) -> tuple[int, str] | None:
    """parse mod:{index}:{param} focus keys"""
    if not key.startswith("mod:"):
        return None
    parts: list[str] = key.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def focus_modifier_field(
    ui: UiState,
    group: Node,
    field: ParamField,
    focus_key: str,
    value: ParamValue,
) -> None:
    """begin typed editing for one modifier param"""
    if field.value_type in ("bool", "enum"):
        return
    ui.focus = FieldId(panel=modifiers_panel, key=focus_key, owner=group.id)
    ui.edit = begin_edit(format_param(field, value))


def commit_modifiers_draft(scene: Scene, ui: UiState, group: Node) -> None:
    """apply the typed draft to its modifier param"""
    if ui.focus is None or ui.edit is None:
        return
    generator: Generator | None = group.generator
    if generator is None:
        return
    mod_ref: tuple[int, str] | None = parse_mod_focus_key(ui.focus.key)
    if mod_ref is None:
        return
    index: int
    param_key: str
    index, param_key = mod_ref
    if index < 0 or index >= len(generator.modifiers):
        return
    field: ParamField = modifier_field_for(generator.modifiers[index].kind, param_key)
    parsed: ParamValue | None = parse_param(field, ui.edit.text)
    if parsed is None:
        return
    set_modifier_param(scene, group.id, index, param_key, parsed)


def focus_modifier_key(scene: Scene, ui: UiState, group_id: str, key: str) -> None:
    """move editing to another modifier param of the same group"""
    node: Node | None = scene.nodes.get(group_id)
    generator: Generator | None = node.generator if node is not None else None
    if node is None or generator is None:
        ui.clear_focus()
        return
    mod_ref: tuple[int, str] | None = parse_mod_focus_key(key)
    if mod_ref is None:
        ui.clear_focus()
        return
    index: int
    param_key: str
    index, param_key = mod_ref
    if index < 0 or index >= len(generator.modifiers):
        ui.clear_focus()
        return
    modifier: Modifier = generator.modifiers[index]
    field: ParamField = modifier_field_for(modifier.kind, param_key)
    value: ParamValue = modifier_params_with_defaults(modifier.kind, modifier.params)[
        param_key
    ]
    focus_modifier_field(ui, node, field, key, value)


def handle_modifiers_typing(
    scene: Scene,
    ui: UiState,
    group: Node,
    keys: Sequence[str],
) -> None:
    """commit on Enter, cancel on Escape, commit and step focus on Tab"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None:
        return
    if ui.focus.panel != modifiers_panel or ui.focus.owner != group.id:
        return
    action: TextAction = handle_text_keys(edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    next_key: str | None = None
    if action is TextAction.focus_next:
        next_key = field_after(keys, ui.focus.key, 1)
    elif action is TextAction.focus_prev:
        next_key = field_after(keys, ui.focus.key, -1)
    commit_modifiers_draft(scene, ui, group)
    if next_key is None:
        ui.clear_focus()
        return
    focus_modifier_key(scene, ui, group.id, next_key)


def update_modifiers_panel(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    group: Node,
) -> tuple[list[Button], list[ParamRowRects]]:
    """handle modifiers panel input; return buttons and row geometry"""
    generator: Generator | None = group.generator
    if generator is None:
        return [], []

    text_keys: list[str] = []
    mod_index: int
    mod_entry: Modifier
    for mod_index, mod_entry in enumerate(generator.modifiers):
        if not known_modifier_kind(mod_entry.kind):
            continue
        mod_field: ParamField
        for mod_field in get_modifier_spec(mod_entry.kind).fields:
            if mod_field.value_type not in ("bool", "enum"):
                text_keys.append(f"mod:{mod_index}:{mod_field.key}")
    handle_modifiers_typing(scene, ui, group, text_keys)

    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    buttons: list[Button] = []
    rows: list[ParamRowRects] = []
    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)

    def on_add_noise() -> None:
        ui.clear_focus()
        set_generator_modifiers(
            scene,
            group.id,
            generator.modifiers + (make_modifier("noise_displace"),),
        )

    buttons.append(
        Button(
            label="+ Noise displace",
            on_click=on_add_noise,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    index: int
    modifier: Modifier
    for index, modifier in enumerate(generator.modifiers):
        if not known_modifier_kind(modifier.kind):
            continue
        mod_spec: ModifierSpec = get_modifier_spec(modifier.kind)
        label: str = mod_spec.label

        def toggle_enabled(i: int = index, current: Modifier = modifier) -> None:
            ui.clear_focus()
            updated: list[Modifier] = list(generator.modifiers)
            updated[i] = replace(current, enabled=not current.enabled)
            set_generator_modifiers(scene, group.id, tuple(updated))

        def move_up(i: int = index) -> None:
            ui.clear_focus()
            if i <= 0:
                return
            updated = list(generator.modifiers)
            updated[i - 1], updated[i] = updated[i], updated[i - 1]
            set_generator_modifiers(scene, group.id, tuple(updated))

        def move_down(i: int = index) -> None:
            ui.clear_focus()
            if i >= len(generator.modifiers) - 1:
                return
            updated = list(generator.modifiers)
            updated[i + 1], updated[i] = updated[i], updated[i + 1]
            set_generator_modifiers(scene, group.id, tuple(updated))

        def remove_mod(i: int = index) -> None:
            ui.clear_focus()
            updated = list(generator.modifiers)
            del updated[i]
            set_generator_modifiers(scene, group.id, tuple(updated))

        enable_rect: Rectangle = Rectangle(x, y, inner_w, row_h)
        rows.append(
            ParamRowRects(
                key=f"modenable:{index}:{label}",
                label=enable_rect,
                minus=Rectangle(0.0, 0.0, 0.0, 0.0),
                value=enable_rect,
                plus=Rectangle(0.0, 0.0, 0.0, 0.0),
                is_bool=True,
                modifier_index=index,
            )
        )
        if clicked and is_point_in_rect(mouse.x, mouse.y, enable_rect):
            toggle_enabled()
        y += row_h + ui_button_gap

        up_rect: Rectangle = Rectangle(x, y, step_w, row_h)
        down_rect: Rectangle = Rectangle(
            x + step_w + ui_button_gap, y, step_w, row_h
        )
        remove_rect: Rectangle = Rectangle(
            x + (step_w + ui_button_gap) * 2, y, step_w, row_h
        )
        buttons.append(Button(label="^", on_click=move_up, rect=up_rect))
        buttons.append(Button(label="v", on_click=move_down, rect=down_rect))
        buttons.append(Button(label="x", on_click=remove_mod, rect=remove_rect))
        y += row_h + ui_button_gap

        params: dict[str, ParamValue] = modifier_params_with_defaults(
            modifier.kind, modifier.params
        )
        field: ParamField
        for field in mod_spec.fields:
            focus_key: str = f"mod:{index}:{field.key}"
            row: ParamRowRects
            row, y = layout_field_row(field, x, y, inner_w)
            row = replace(row, key=focus_key, modifier_index=index)
            rows.append(row)

            if row.is_bool:
                if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                    set_modifier_param(
                        scene,
                        group.id,
                        index,
                        field.key,
                        0 if int(params[field.key]) else 1,
                    )
                continue
            if row.is_enum:
                if clicked:
                    options: tuple[tuple[int, str], ...] = field.options or ()
                    option_index: int
                    option_rect: Rectangle
                    for option_index, option_rect in enumerate(row.option_rects):
                        if is_point_in_rect(mouse.x, mouse.y, option_rect):
                            ui.clear_focus()
                            set_modifier_param(
                                scene,
                                group.id,
                                index,
                                field.key,
                                options[option_index][0],
                            )
                            break
                continue

            def step_minus(i: int = index, k: str = field.key) -> None:
                ui.clear_focus()
                step_modifier_param(scene, group.id, i, k, -1)

            def step_plus(i: int = index, k: str = field.key) -> None:
                ui.clear_focus()
                step_modifier_param(scene, group.id, i, k, 1)

            buttons.append(Button(label="-", on_click=step_minus, rect=row.minus))
            buttons.append(Button(label="+", on_click=step_plus, rect=row.plus))
            if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                focus_modifier_field(ui, group, field, focus_key, params[field.key])

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
            ui.clear_focus()

    return buttons, rows


def draw_modifiers_panel(
    font: Font,
    area: Rectangle,
    ui: UiState,
    group: Node,
    buttons: list[Button],
    rows: list[ParamRowRects],
) -> None:
    """draw the modifiers column for a parametric group"""
    generator: Generator | None = group.generator
    if generator is None:
        return
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        "Modifiers",
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    row: ParamRowRects
    for row in rows:
        if row.key.startswith("modenable:"):
            enable_label: str = (
                row.key.split(":", 2)[2] if row.key.count(":") >= 2 else "Modifier"
            )
            checked: bool = False
            if (
                row.modifier_index is not None
                and 0 <= row.modifier_index < len(generator.modifiers)
            ):
                checked = generator.modifiers[row.modifier_index].enabled
            draw_checkbox_with_label(font, row.value, enable_label, checked=checked)
            continue

        mod_ref: tuple[int, str] | None = parse_mod_focus_key(row.key)
        if mod_ref is None:
            continue
        mod_i: int
        param_key: str
        mod_i, param_key = mod_ref
        if mod_i < 0 or mod_i >= len(generator.modifiers):
            continue
        modifier: Modifier = generator.modifiers[mod_i]
        if not known_modifier_kind(modifier.kind):
            continue
        field: ParamField = modifier_field_for(modifier.kind, param_key)
        value: ParamValue = modifier_params_with_defaults(
            modifier.kind, modifier.params
        )[param_key]
        if row.is_bool:
            draw_checkbox_with_label(
                font,
                row.value,
                field.label,
                checked=bool(int(value)),
            )
            continue
        draw_text_ex(
            font,
            field.label,
            Vector2(row.label.x, row.label.y),
            float(ui_font_size),
            0,
            ui_color_text,
        )
        if row.is_enum:
            options: tuple[tuple[int, str], ...] = field.options or ()
            option_index: int
            option_rect: Rectangle
            for option_index, option_rect in enumerate(row.option_rects):
                option_value: int
                option_label: str
                option_value, option_label = options[option_index]
                draw_radio_option(
                    font,
                    option_rect,
                    option_label,
                    selected=int(value) == option_value,
                )
            continue
        edit: TextEdit | None = ui.focused_edit(modifiers_panel, row.key, group.id)
        if edit is not None:
            draw_text_field(
                font,
                row.value,
                edit.text,
                focused=True,
                caret=edit.caret,
                mark=edit.mark,
            )
            continue
        draw_text_field(
            font,
            row.value,
            format_param(field, value),
            focused=False,
            center_unfocused=True,
        )

    button: Button
    for button in buttons:
        draw_button(button, font)
