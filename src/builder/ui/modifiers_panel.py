"""modifiers column: stack editor for the selected generator group"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any

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

from builder.generators.generator_types import Generator, Modifier
from builder.generators.regenerate import (
    set_generator_modifiers,
    set_modifier_params,
)
from builder.modifiers.modifier_types import ModifierSpec
from builder.modifiers.registry import (
    get_modifier_spec,
    known_modifier_kind,
    make_modifier,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.param_ui import (
    BoolControl,
    EnumControl,
    Float3ComponentControl,
    FloatControl,
    Int3ComponentControl,
    IntControl,
    ParamControl,
    ParamRowRects,
    control_by_key,
    control_key,
    control_label,
    control_step,
    controls_from_params_type,
    enum_members,
    format_control,
    layout_control_row,
    parse_control,
    read_control,
    write_control,
)
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


@dataclass(frozen=True)
class ModifierEnableRow:
    """hit target for one modifier enable checkbox"""

    key: str
    rect: Rectangle
    modifier_index: int
    label: str


@dataclass(frozen=True)
class ModifierParamRow:
    """one reflected param control inside a modifier entry"""

    focus_key: str
    modifier_index: int
    control: ParamControl
    row: ParamRowRects


type ModifierPanelRow = ModifierEnableRow | ModifierParamRow


def sync_modifiers_focus(ui: UiState, group: Node | None) -> None:
    """clear draft editing when the selected generator changes"""
    if ui.focus is None or ui.focus.panel != modifiers_panel:
        return
    if group is None or ui.focus.owner != group.id:
        ui.clear_focus()


def parse_mod_focus_key(key: str) -> tuple[int, str] | None:
    """parse mod:{index}:{control_key} focus keys"""
    if not key.startswith("mod:"):
        return None
    parts: list[str] = key.split(":", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def focus_modifier_control(
    ui: UiState,
    group: Node,
    control: ParamControl,
    focus_key: str,
    params: Any,
) -> None:
    """begin typed editing for one modifier control"""
    match control:
        case BoolControl() | EnumControl():
            return
        case _:
            pass
    ui.focus = FieldId(panel=modifiers_panel, key=focus_key, owner=group.id)
    ui.edit = begin_edit(format_control(control, read_control(params, control)))


def commit_modifiers_draft(scene: Scene, ui: UiState, group: Node) -> None:
    """apply the typed draft to its modifier control"""
    if ui.focus is None or ui.edit is None:
        return
    generator: Generator | None = group.generator
    if generator is None:
        return
    mod_ref: tuple[int, str] | None = parse_mod_focus_key(ui.focus.key)
    if mod_ref is None:
        return
    index: int
    control_id: str
    index, control_id = mod_ref
    if index < 0 or index >= len(generator.modifiers):
        return
    modifier: Modifier = generator.modifiers[index]
    if not known_modifier_kind(modifier.kind):
        return
    controls: tuple[ParamControl, ...] = controls_from_params_type(
        get_modifier_spec(modifier.kind).params_type
    )
    control: ParamControl = control_by_key(controls, control_id)
    parsed: Any | None = parse_control(control, ui.edit.text)
    if parsed is None:
        return
    set_modifier_params(
        scene, group.id, index, write_control(modifier.params, control, parsed)
    )


def focus_modifier_key(scene: Scene, ui: UiState, group_id: str, key: str) -> None:
    """move editing to another modifier control of the same group"""
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
    control_id: str
    index, control_id = mod_ref
    if index < 0 or index >= len(generator.modifiers):
        ui.clear_focus()
        return
    modifier: Modifier = generator.modifiers[index]
    if not known_modifier_kind(modifier.kind):
        ui.clear_focus()
        return
    controls: tuple[ParamControl, ...] = controls_from_params_type(
        get_modifier_spec(modifier.kind).params_type
    )
    control: ParamControl = control_by_key(controls, control_id)
    focus_modifier_control(ui, node, control, key, modifier.params)


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
) -> tuple[list[Button], list[ModifierPanelRow]]:
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
        control: ParamControl
        for control in controls_from_params_type(
            get_modifier_spec(mod_entry.kind).params_type
        ):
            match control:
                case BoolControl() | EnumControl():
                    continue
                case _:
                    text_keys.append(f"mod:{mod_index}:{control_key(control)}")
    handle_modifiers_typing(scene, ui, group, text_keys)

    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    buttons: list[Button] = []
    rows: list[ModifierPanelRow] = []
    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)

    def on_add_noise() -> None:
        ui.clear_focus()
        set_generator_modifiers(
            scene,
            group.id,
            generator.modifiers + (make_modifier("noise_displace"),),
        )

    def on_add_translate() -> None:
        ui.clear_focus()
        set_generator_modifiers(
            scene,
            group.id,
            generator.modifiers + (make_modifier("template_translate"),),
        )

    buttons.append(
        Button(
            label="+ Noise displace",
            on_click=on_add_noise,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Translate (template)",
            on_click=on_add_translate,
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
            ModifierEnableRow(
                key=f"modenable:{index}",
                rect=enable_rect,
                modifier_index=index,
                label=mod_spec.label,
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

        controls: tuple[ParamControl, ...] = controls_from_params_type(
            mod_spec.params_type
        )
        for control in controls:
            focus_key: str = f"mod:{index}:{control_key(control)}"
            layout_row: ParamRowRects
            layout_row, y = layout_control_row(
                control, modifier.params, x, y, inner_w
            )
            rows.append(
                ModifierParamRow(
                    focus_key=focus_key,
                    modifier_index=index,
                    control=control,
                    row=replace(layout_row, key=focus_key),
                )
            )

            match control:
                case BoolControl() as bool_control:
                    if clicked and is_point_in_rect(
                        mouse.x, mouse.y, layout_row.value
                    ):
                        current_bool: bool = bool(
                            read_control(modifier.params, bool_control)
                        )
                        set_modifier_params(
                            scene,
                            group.id,
                            index,
                            write_control(
                                modifier.params, bool_control, not current_bool
                            ),
                        )
                case EnumControl() as enum_control:
                    if clicked:
                        options: tuple[IntEnum, ...] = enum_members(
                            modifier.params, enum_control
                        )
                        option_index: int
                        option_rect: Rectangle
                        for option_index, option_rect in enumerate(
                            layout_row.option_rects
                        ):
                            if is_point_in_rect(mouse.x, mouse.y, option_rect):
                                ui.clear_focus()
                                set_modifier_params(
                                    scene,
                                    group.id,
                                    index,
                                    write_control(
                                        modifier.params,
                                        enum_control,
                                        options[option_index],
                                    ),
                                )
                                break
                case (
                    IntControl()
                    | FloatControl()
                    | Float3ComponentControl()
                    | Int3ComponentControl()
                ) as stepped:
                    def step_minus(
                        i: int = index,
                        c: ParamControl = stepped,
                        p: Any = modifier.params,
                    ) -> None:
                        ui.clear_focus()
                        current: Any = read_control(p, c)
                        set_modifier_params(
                            scene,
                            group.id,
                            i,
                            write_control(
                                p,
                                c,
                                float(current) - control_step(p, c),
                            ),
                        )

                    def step_plus(
                        i: int = index,
                        c: ParamControl = stepped,
                        p: Any = modifier.params,
                    ) -> None:
                        ui.clear_focus()
                        current = read_control(p, c)
                        set_modifier_params(
                            scene,
                            group.id,
                            i,
                            write_control(
                                p,
                                c,
                                float(current) + control_step(p, c),
                            ),
                        )

                    buttons.append(
                        Button(label="-", on_click=step_minus, rect=layout_row.minus)
                    )
                    buttons.append(
                        Button(label="+", on_click=step_plus, rect=layout_row.plus)
                    )
                    if clicked and is_point_in_rect(
                        mouse.x, mouse.y, layout_row.value
                    ):
                        focus_modifier_control(
                            ui, group, stepped, focus_key, modifier.params
                        )

    update_buttons(buttons, area)

    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_control: bool = False
        button: Button
        for button in buttons:
            if is_point_in_rect(mouse.x, mouse.y, button.rect):
                on_control = True
                break
        on_value: bool = False
        panel_row: ModifierPanelRow
        for panel_row in rows:
            match panel_row:
                case ModifierEnableRow(rect=rect):
                    if is_point_in_rect(mouse.x, mouse.y, rect):
                        on_value = True
                case ModifierParamRow(row=row):
                    if is_point_in_rect(mouse.x, mouse.y, row.value):
                        on_value = True
                    option_rect: Rectangle
                    for option_rect in row.option_rects:
                        if is_point_in_rect(mouse.x, mouse.y, option_rect):
                            on_value = True
        if not on_control and not on_value:
            ui.clear_focus()

    return buttons, rows


def draw_modifiers_panel(
    font: Font,
    area: Rectangle,
    ui: UiState,
    group: Node,
    buttons: list[Button],
    rows: list[ModifierPanelRow],
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
    panel_row: ModifierPanelRow
    for panel_row in rows:
        match panel_row:
            case ModifierEnableRow(
                rect=rect, modifier_index=mod_i, label=enable_label
            ):
                checked: bool = False
                if 0 <= mod_i < len(generator.modifiers):
                    checked = generator.modifiers[mod_i].enabled
                draw_checkbox_with_label(
                    font, rect, enable_label, checked=checked
                )
            case ModifierParamRow(
                focus_key=focus_key,
                modifier_index=mod_i,
                control=control,
                row=row,
            ):
                if mod_i < 0 or mod_i >= len(generator.modifiers):
                    continue
                modifier: Modifier = generator.modifiers[mod_i]
                if not known_modifier_kind(modifier.kind):
                    continue
                value: Any = read_control(modifier.params, control)
                match control:
                    case BoolControl():
                        draw_checkbox_with_label(
                            font,
                            row.value,
                            control_label(modifier.params, control),
                            checked=bool(value),
                        )
                    case EnumControl() as enum_control:
                        draw_text_ex(
                            font,
                            control_label(modifier.params, control),
                            Vector2(row.label.x, row.label.y),
                            float(ui_font_size),
                            0,
                            ui_color_text,
                        )
                        options: tuple[IntEnum, ...] = enum_members(
                            modifier.params, enum_control
                        )
                        option_index: int
                        option_rect: Rectangle
                        for option_index, option_rect in enumerate(row.option_rects):
                            member: IntEnum = options[option_index]
                            draw_radio_option(
                                font,
                                option_rect,
                                member.name.capitalize(),
                                selected=int(value) == int(member),
                            )
                    case _:
                        draw_text_ex(
                            font,
                            control_label(modifier.params, control),
                            Vector2(row.label.x, row.label.y),
                            float(ui_font_size),
                            0,
                            ui_color_text,
                        )
                        edit: TextEdit | None = ui.focused_edit(
                            modifiers_panel, focus_key, group.id
                        )
                        if edit is not None:
                            draw_text_field(
                                font,
                                row.value,
                                edit.text,
                                focused=True,
                                caret=edit.caret,
                                mark=edit.mark,
                            )
                        else:
                            draw_text_field(
                                font,
                                row.value,
                                format_control(control, value),
                                focused=False,
                                center_unfocused=True,
                            )

    button: Button
    for button in buttons:
        draw_button(button, font)
