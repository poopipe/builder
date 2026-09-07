"""right-side inspector for parametric group recipes"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from random import randrange
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

from builder.generators.generator_types import (
    Generator,
    GeneratorSpec,
    MeshPattern,
    MeshSequenceMode,
)
from builder.generators.registry import get_spec
from builder.generators.mesh_pattern import next_mode, remove_slot, set_slot
from builder.generators.regenerate import (
    bake_group,
    set_generator_params,
    set_generator_pattern,
)
from builder.ui.param_ui import (
    BoolControl,
    EnumControl,
    FloatControl,
    IntControl,
    Float3ComponentControl,
    Int3ComponentControl,
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
from builder.generators.spline_edit import add_spline_point
from builder.meshes.mesh_catalog import MeshAsset, MeshCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
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


inspector_panel: str = "inspector"


def apply_control_edit(
    scene: Scene,
    group_id: str,
    generator: Generator,
    control: ParamControl,
    value: Any,
) -> None:
    """write one inspector control into params and regenerate"""
    set_generator_params(
        scene, group_id, write_control(generator.params, control, value)
    )


def sync_inspector_focus(ui: UiState, group: Node | None) -> None:
    """clear draft editing when the inspected group changes"""
    if ui.focus is None or ui.focus.panel != inspector_panel:
        return
    if group is None or ui.focus.owner != group.id:
        ui.clear_focus()


group_header_height: float = float(ui_button_height)


@dataclass(frozen=True)
class GroupHeaderRect:
    """collapse toggle for a contiguous run of grouped params"""

    title: str
    rect: Rectangle
    expanded: bool


@dataclass(frozen=True)
class ParamLayout:
    """laid-out params: group headers, visible rows, and content bottom edge"""

    headers: tuple[GroupHeaderRect, ...]
    rows: tuple[ParamRowRects, ...]
    bottom: float


def layout_params(
    area: Rectangle, controls: tuple[ParamControl, ...], params: Any
) -> ParamLayout:
    """place inspector rows for reflected params controls"""
    rows: list[ParamRowRects] = []
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    control: ParamControl
    for control in controls:
        row: ParamRowRects
        row, y = layout_control_row(control, params, x, y, inner_w)
        rows.append(row)
    return ParamLayout((), tuple(rows), y)


def toggle_inspector_group(ui: UiState, title: str) -> None:
    """expand or collapse an inspector param group"""
    if title in ui.inspector_collapsed:
        ui.inspector_collapsed.discard(title)
    else:
        ui.inspector_collapsed.add(title)


def append_group_header(
    buttons: list[Button],
    ui: UiState,
    area: Rectangle,
    y: float,
    title: str,
) -> tuple[Rectangle, float, bool]:
    """append a collapse header button; return (rect, next_y, expanded)"""
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    rect: Rectangle = Rectangle(x, y, inner_w, group_header_height)
    expanded: bool = title not in ui.inspector_collapsed

    def on_toggle(name: str = title) -> None:
        ui.clear_focus()
        toggle_inspector_group(ui, name)

    caret: str = "v" if expanded else ">"
    buttons.append(
        Button(
            label=f"{caret}  {title}",
            on_click=on_toggle,
            rect=rect,
            align_left=True,
        )
    )
    return rect, y + group_header_height + ui_button_gap, expanded


def focus_param_control(
    ui: UiState,
    group: Node,
    control: ParamControl,
) -> None:
    """begin typed editing for one numeric generator control"""
    match control:
        case BoolControl() | EnumControl():
            return
        case _:
            pass
    generator: Generator | None = group.generator
    if generator is None:
        return
    value: Any = read_control(generator.params, control)
    ui.focus = FieldId(
        panel=inspector_panel, key=control_key(control), owner=group.id
    )
    ui.edit = begin_edit(format_control(control, value))


def commit_inspector_draft(
    scene: Scene,
    ui: UiState,
    group: Node,
    controls: tuple[ParamControl, ...],
) -> None:
    """apply the typed draft to its control; invalid text is discarded"""
    if ui.focus is None or ui.edit is None:
        return
    generator: Generator | None = group.generator
    if generator is None:
        return
    control: ParamControl = control_by_key(controls, ui.focus.key)
    parsed: Any | None = parse_control(control, ui.edit.text)
    if parsed is None:
        return
    apply_control_edit(scene, group.id, generator, control, parsed)


def focus_param_key(
    scene: Scene,
    ui: UiState,
    group_id: str,
    key: str,
    controls: tuple[ParamControl, ...],
) -> None:
    """move editing to another control of the same group"""
    node: Node | None = scene.nodes.get(group_id)
    generator: Generator | None = node.generator if node is not None else None
    if node is None or generator is None:
        ui.clear_focus()
        return
    focus_param_control(ui, node, control_by_key(controls, key))


def handle_inspector_typing(
    scene: Scene,
    ui: UiState,
    group: Node,
    keys: Sequence[str],
    controls: tuple[ParamControl, ...],
) -> None:
    """commit on Enter, cancel on Escape, commit and step focus on Tab"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None:
        return
    if ui.focus.panel != inspector_panel or ui.focus.owner != group.id:
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
    commit_inspector_draft(scene, ui, group, controls)
    if next_key is None:
        ui.clear_focus()
        return
    focus_param_key(scene, ui, group.id, next_key, controls)


def mesh_label(catalog: MeshCatalog, mesh_id: MeshId) -> str:
    """return the catalog label for a mesh id, or its raw name if unknown"""
    asset: MeshAsset | None = catalog.entries.get(mesh_id)
    return asset.label if asset is not None else mesh_id.name


def build_pattern_buttons(
    scene: Scene,
    ui: UiState,
    group: Node,
    pattern: MeshPattern,
    catalog: MeshCatalog,
    active_mesh_id: MeshId,
    area: Rectangle,
    start_y: float,
    *,
    points: bool = False,
) -> tuple[list[Button], float]:
    """build mesh-pattern controls below the param rows; return (buttons, next_y)"""
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    y: float = start_y
    buttons: list[Button] = []

    def clear() -> None:
        ui.clear_focus()

    def cycle_mode() -> None:
        clear()
        set_generator_pattern(
            scene,
            group.id,
            replace(pattern, mode=next_mode(pattern.mode)),
            points=points,
        )

    buttons.append(
        Button(
            label=f"Mode: {pattern.mode.name}",
            on_click=cycle_mode,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    match pattern.mode:
        case MeshSequenceMode.random:

            def seed_minus() -> None:
                clear()
                set_generator_pattern(
                    scene,
                    group.id,
                    replace(pattern, seed=max(0, pattern.seed - 1)),
                    points=points,
                )

            def seed_plus() -> None:
                clear()
                set_generator_pattern(
                    scene,
                    group.id,
                    replace(pattern, seed=pattern.seed + 1),
                    points=points,
                )

            def seed_reroll() -> None:
                clear()
                set_generator_pattern(
                    scene,
                    group.id,
                    replace(pattern, seed=randrange(0, 1_000_000)),
                    points=points,
                )

            value_x: float = x + step_w + ui_button_gap
            value_w: float = inner_w - (step_w + ui_button_gap) * 2
            buttons.append(
                Button(
                    label="-",
                    on_click=seed_minus,
                    rect=Rectangle(x, y, step_w, row_h),
                )
            )
            buttons.append(
                Button(
                    label=f"Seed {pattern.seed}",
                    on_click=seed_reroll,
                    rect=Rectangle(value_x, y, value_w, row_h),
                )
            )
            buttons.append(
                Button(
                    label="+",
                    on_click=seed_plus,
                    rect=Rectangle(x + inner_w - step_w, y, step_w, row_h),
                )
            )
            y += row_h + ui_button_gap
        case MeshSequenceMode.repeat | MeshSequenceMode.pingpong:
            pass

    slot: int
    mesh_id: MeshId
    for slot, mesh_id in enumerate(pattern.mesh_ids):

        def assign(s: int = slot) -> None:
            clear()
            set_generator_pattern(
                scene, group.id, set_slot(pattern, s, active_mesh_id), points=points
            )

        def remove(s: int = slot) -> None:
            clear()
            set_generator_pattern(
                scene, group.id, remove_slot(pattern, s), points=points
            )

        buttons.append(
            Button(
                label=f"{slot + 1}. {mesh_label(catalog, mesh_id)}",
                on_click=assign,
                rect=Rectangle(x, y, inner_w - step_w - ui_button_gap, row_h),
            )
        )
        buttons.append(
            Button(
                label="x",
                on_click=remove,
                rect=Rectangle(x + inner_w - step_w, y, step_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    def add_active() -> None:
        clear()
        set_generator_pattern(
            scene,
            group.id,
            replace(pattern, mesh_ids=pattern.mesh_ids + (active_mesh_id,)),
            points=points,
        )

    buttons.append(
        Button(
            label="+ Add active mesh",
            on_click=add_active,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_pad
    return buttons, y


def update_inspector(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    group: Node,
    catalog: MeshCatalog,
    active_mesh_id: MeshId,
) -> tuple[list[Button], list[ParamRowRects]]:
    """handle inspector input; return stepper/bake buttons and row geometry"""
    generator: Generator | None = group.generator
    if generator is None:
        return [], []
    spec: GeneratorSpec = get_spec(generator.kind)
    controls: tuple[ParamControl, ...] = controls_from_params_type(spec.params_type)
    layout: ParamLayout = layout_params(area, controls, generator.params)
    rows: list[ParamRowRects] = list(layout.rows)
    text_keys: list[str] = [
        row.key
        for row in rows
        if not isinstance(
            control_by_key(controls, row.key), (BoolControl, EnumControl)
        )
    ]
    handle_inspector_typing(scene, ui, group, text_keys, controls)

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    buttons: list[Button] = []
    header: GroupHeaderRect
    for header in layout.headers:

        def toggle_group(title: str = header.title) -> None:
            ui.clear_focus()
            toggle_inspector_group(ui, title)

        caret: str = "v" if header.expanded else ">"
        buttons.append(
            Button(
                label=f"{caret}  {header.title}",
                on_click=toggle_group,
                rect=header.rect,
                align_left=True,
            )
        )
    row: ParamRowRects
    for row in rows:
        control: ParamControl = control_by_key(controls, row.key)
        match control:
            case BoolControl() as bool_control:
                current_bool: bool = bool(
                    read_control(generator.params, bool_control)
                )

                def toggle_bool(
                    c: BoolControl = bool_control, current: bool = current_bool
                ) -> None:
                    ui.clear_focus()
                    apply_control_edit(
                        scene, group.id, generator, c, not current
                    )

                if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                    toggle_bool()
            case EnumControl() as enum_control:
                if clicked:
                    options: tuple[IntEnum, ...] = enum_members(
                        generator.params, enum_control
                    )
                    option_index: int
                    option_rect: Rectangle
                    for option_index, option_rect in enumerate(row.option_rects):
                        if is_point_in_rect(mouse.x, mouse.y, option_rect):
                            ui.clear_focus()
                            apply_control_edit(
                                scene,
                                group.id,
                                generator,
                                enum_control,
                                options[option_index],
                            )
                            break
            case (
                IntControl()
                | FloatControl()
                | Float3ComponentControl()
                | Int3ComponentControl()
            ) as stepped:
                step: float = control_step(generator.params, stepped)

                def step_minus(
                    c: ParamControl = stepped, s: float = step
                ) -> None:
                    ui.clear_focus()
                    current: Any = read_control(generator.params, c)
                    apply_control_edit(
                        scene, group.id, generator, c, float(current) - s
                    )

                def step_plus(
                    c: ParamControl = stepped, s: float = step
                ) -> None:
                    ui.clear_focus()
                    current: Any = read_control(generator.params, c)
                    apply_control_edit(
                        scene, group.id, generator, c, float(current) + s
                    )

                buttons.append(
                    Button(label="-", on_click=step_minus, rect=row.minus)
                )
                buttons.append(
                    Button(label="+", on_click=step_plus, rect=row.plus)
                )
                if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                    focus_param_control(ui, group, stepped)

    pattern_start: float = layout.bottom
    bake_y: float = pattern_start
    edge_title: str = (
        "Edge meshes" if spec.supports_point_meshes else "Meshes"
    )
    _: Rectangle
    edge_open: bool
    _, bake_y, edge_open = append_group_header(
        buttons, ui, area, bake_y, edge_title
    )
    if edge_open:
        pattern_buttons: list[Button]
        pattern_buttons, bake_y = build_pattern_buttons(
            scene,
            ui,
            group,
            generator.meshes,
            catalog,
            active_mesh_id,
            area,
            bake_y,
            points=False,
        )
        buttons.extend(pattern_buttons)
    if spec.supports_point_meshes:
        point_open: bool
        _, bake_y, point_open = append_group_header(
            buttons, ui, area, bake_y, "Point meshes"
        )
        if point_open:
            point_pattern: MeshPattern = (
                generator.point_meshes
                if generator.point_meshes is not None
                else generator.meshes
            )
            point_buttons: list[Button]
            point_buttons, bake_y = build_pattern_buttons(
                scene,
                ui,
                group,
                point_pattern,
                catalog,
                active_mesh_id,
                area,
                bake_y,
                points=True,
            )
            buttons.extend(point_buttons)

    if generator.kind == "spline":

        def on_add_point() -> None:
            ui.clear_focus()
            add_spline_point(scene, group.id)
            ui.status = "Added spline point"

        buttons.append(
            Button(
                label="Add point",
                on_click=on_add_point,
                rect=Rectangle(
                    area.x + ui_pad,
                    bake_y,
                    area.width - ui_pad * 2,
                    float(ui_button_height),
                ),
            )
        )
        bake_y += float(ui_button_height) + ui_pad

    bake_rect: Rectangle = Rectangle(
        area.x + ui_pad,
        bake_y,
        area.width - ui_pad * 2,
        float(ui_button_height),
    )

    def on_bake() -> None:
        ui.clear_focus()
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
            ui.clear_focus()

    return buttons, rows


def draw_inspector(
    font: Font,
    area: Rectangle,
    ui: UiState,
    group: Node,
    buttons: list[Button],
    rows: list[ParamRowRects],
) -> None:
    """draw the right inspector panel for a parametric group"""
    generator: Generator | None = group.generator
    if generator is None:
        return
    spec: GeneratorSpec = get_spec(generator.kind)
    controls: tuple[ParamControl, ...] = controls_from_params_type(spec.params_type)
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        spec.label,
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    row: ParamRowRects
    for row in rows:
        control: ParamControl = control_by_key(controls, row.key)
        value: Any = read_control(generator.params, control)
        match control:
            case BoolControl():
                draw_checkbox_with_label(
                    font,
                    row.value,
                    control_label(generator.params, control),
                    checked=bool(value),
                )
            case EnumControl() as enum_control:
                draw_text_ex(
                    font,
                    control_label(generator.params, control),
                    Vector2(row.label.x, row.label.y),
                    float(ui_font_size),
                    0,
                    ui_color_text,
                )
                options: tuple[IntEnum, ...] = enum_members(
                    generator.params, enum_control
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
            case IntControl() | FloatControl() | Float3ComponentControl() | Int3ComponentControl():
                draw_text_ex(
                    font,
                    control_label(generator.params, control),
                    Vector2(row.label.x, row.label.y),
                    float(ui_font_size),
                    0,
                    ui_color_text,
                )
                edit: TextEdit | None = ui.focused_edit(
                    inspector_panel, row.key, group.id
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
