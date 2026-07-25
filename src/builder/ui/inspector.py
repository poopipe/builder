"""right-side inspector for parametric group recipes"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import randrange

from pyray import (
    Color,
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

from builder.generators.generator_types import (
    Generator,
    GeneratorSpec,
    MeshPattern,
    ParamField,
    ParamValue,
)
from builder.generators.registry import (
    field_for,
    format_param,
    get_spec,
    params_with_defaults,
    parse_param,
)
from builder.generators.mesh_pattern import next_mode, remove_slot, set_slot
from builder.generators.regenerate import (
    bake_group,
    set_generator_param,
    set_generator_pattern,
    step_generator_param,
)
from builder.meshes.mesh_catalog import MeshAsset, MeshCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_button,
    ui_color_input,
    ui_color_input_focus,
    ui_color_panel,
    ui_color_selection,
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


@dataclass(frozen=True)
class ParamRowRects:
    """hit targets for one inspector param row"""

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def sync_inspector_focus(ui: UiState, group: Node | None) -> None:
    """clear draft editing when the inspected group changes"""
    if group is None:
        ui.clear_inspector_focus()
        return
    if ui.inspector_group_id is not None and ui.inspector_group_id != group.id:
        ui.clear_inspector_focus()


def apply_inspector_exit_key(ui: UiState) -> None:
    """disable window-close-on-escape while typing or a file dialog is open"""
    if ui.inspector_focus_key is not None or ui.file_browser is not None:
        set_exit_key(0)
    else:
        set_exit_key(KeyboardKey.KEY_ESCAPE)


def layout_param_rows(area: Rectangle, fields: tuple[ParamField, ...]) -> list[ParamRowRects]:
    """place label / stepper / value rows under the inspector title"""
    rows: list[ParamRowRects] = []
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    field: ParamField
    for field in fields:
        label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        y += float(ui_font_size) + 4.0
        controls_y: float = y
        minus: Rectangle = Rectangle(
            x,
            controls_y,
            float(ui_stepper_width),
            float(ui_button_height),
        )
        plus: Rectangle = Rectangle(
            x + inner_w - float(ui_stepper_width),
            controls_y,
            float(ui_stepper_width),
            float(ui_button_height),
        )
        value: Rectangle = Rectangle(
            minus.x + minus.width + ui_button_gap,
            controls_y,
            plus.x - (minus.x + minus.width + ui_button_gap * 2),
            float(ui_button_height),
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
        y += float(ui_button_height) + ui_pad
    return rows


def bake_button_rect(area: Rectangle, rows: list[ParamRowRects]) -> Rectangle:
    """place the Bake button below the last param row"""
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    if rows:
        last: ParamRowRects = rows[-1]
        y = last.plus.y + last.plus.height + ui_pad
    return Rectangle(
        area.x + ui_pad,
        y,
        area.width - ui_pad * 2,
        float(ui_button_height),
    )


def focus_param_field(ui: UiState, group: Node, field: ParamField) -> None:
    """begin typed editing for one param"""
    generator: Generator | None = group.generator
    if generator is None:
        return
    value: ParamValue = params_with_defaults(generator.kind, generator.params)[
        field.key
    ]
    ui.inspector_group_id = group.id
    ui.inspector_focus_key = field.key
    ui.inspector_draft = format_param(field, value)
    ui.inspector_select_all = True


def commit_inspector_draft(scene: Scene, ui: UiState, group: Node) -> None:
    """apply typed draft on Enter; invalid text is discarded"""
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
    """typed-field keyboard: commit on Enter, cancel on Escape"""
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
) -> tuple[list[Button], float]:
    """build mesh-pattern controls below the param rows; return (buttons, next_y)"""
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    y: float = start_y
    buttons: list[Button] = []

    def clear() -> None:
        ui.clear_inspector_focus()

    def cycle_mode() -> None:
        clear()
        set_generator_pattern(
            scene, group.id, replace(pattern, mode=next_mode(pattern.mode))
        )

    buttons.append(
        Button(
            label=f"Order: {pattern.mode}",
            on_click=cycle_mode,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    if pattern.mode == "random":

        def seed_minus() -> None:
            clear()
            set_generator_pattern(
                scene, group.id, replace(pattern, seed=max(0, pattern.seed - 1))
            )

        def seed_plus() -> None:
            clear()
            set_generator_pattern(
                scene, group.id, replace(pattern, seed=pattern.seed + 1)
            )

        def seed_reroll() -> None:
            clear()
            set_generator_pattern(
                scene, group.id, replace(pattern, seed=randrange(0, 1_000_000))
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

    slot: int
    mesh_id: MeshId
    for slot, mesh_id in enumerate(pattern.mesh_ids):

        def assign(s: int = slot) -> None:
            clear()
            set_generator_pattern(
                scene, group.id, set_slot(pattern, s, active_mesh_id)
            )

        def remove(s: int = slot) -> None:
            clear()
            set_generator_pattern(scene, group.id, remove_slot(pattern, s))

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

        buttons.append(Button(label="-", on_click=step_minus, rect=row.minus))
        buttons.append(Button(label="+", on_click=step_plus, rect=row.plus))
        if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
            focus_param_field(ui, group, field)

    pattern_start: float = bake_button_rect(area, rows).y
    pattern_buttons: list[Button]
    bake_y: float
    pattern_buttons, bake_y = build_pattern_buttons(
        scene,
        ui,
        group,
        generator.meshes,
        catalog,
        active_mesh_id,
        area,
        pattern_start,
    )
    buttons.extend(pattern_buttons)

    bake_rect: Rectangle = Rectangle(
        area.x + ui_pad,
        bake_y,
        area.width - ui_pad * 2,
        float(ui_button_height),
    )

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
    """draw the editable value box for one param"""
    draw_rectangle_rec(
        rect, ui_color_input if focused else ui_color_button
    )
    border: Color = ui_color_input_focus if focused else ui_color_border
    draw_rectangle_lines_ex(rect, 2.0 if focused else 1.0, border)

    pad_x: float = 6.0
    text_w: float = measure_text_ex(font, text, float(ui_font_size), 0).x
    ty: float = rect.y + (rect.height - float(ui_font_size)) * 0.5
    tx: float
    if focused:
        tx = rect.x + pad_x
    else:
        tx = rect.x + (rect.width - text_w) * 0.5

    if focused and select_all and text != "":
        draw_rectangle_rec(
            Rectangle(
                tx - 1.0,
                ty - 1.0,
                text_w + 2.0,
                float(ui_font_size) + 2.0,
            ),
            ui_color_selection,
        )

    draw_text_ex(
        font, text, Vector2(tx, ty), float(ui_font_size), 0, ui_color_text
    )

    if focused and not select_all and int(get_time() * 2.0) % 2 == 0:
        caret_x: float = tx + text_w + 1.0
        draw_rectangle_rec(
            Rectangle(caret_x, ty, 1.0, float(ui_font_size)),
            ui_color_text,
        )


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
        field: ParamField = field_for(spec, row.key)
        draw_text_ex(
            font,
            field.label,
            Vector2(row.label.x, row.label.y),
            float(ui_font_size),
            0,
            ui_color_text,
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
            display = format_param(
                field,
                params_with_defaults(generator.kind, generator.params)[row.key],
            )
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
