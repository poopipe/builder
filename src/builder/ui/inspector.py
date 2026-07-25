"""right-side inspector for parametric group recipes"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from random import randrange

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
    is_point_in_rect,
    update_buttons,
)


inspector_panel: str = "inspector"


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
    if ui.focus is None or ui.focus.panel != inspector_panel:
        return
    if group is None or ui.focus.owner != group.id:
        ui.clear_focus()


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
    ui.focus = FieldId(panel=inspector_panel, key=field.key, owner=group.id)
    ui.edit = begin_edit(format_param(field, value))


def commit_inspector_draft(scene: Scene, ui: UiState, group: Node) -> None:
    """apply the typed draft to its param; invalid text is discarded"""
    if ui.focus is None or ui.edit is None:
        return
    generator: Generator | None = group.generator
    if generator is None:
        return
    field: ParamField = field_for(get_spec(generator.kind), ui.focus.key)
    parsed: ParamValue | None = parse_param(field, ui.edit.text)
    if parsed is None:
        return
    set_generator_param(scene, group.id, field.key, parsed)


def focus_param_key(scene: Scene, ui: UiState, group_id: str, key: str) -> None:
    """move editing to another param of the same group, reading its current value"""
    node: Node | None = scene.nodes.get(group_id)
    generator: Generator | None = node.generator if node is not None else None
    if node is None or generator is None:
        ui.clear_focus()
        return
    focus_param_field(ui, node, field_for(get_spec(generator.kind), key))


def handle_inspector_typing(
    scene: Scene,
    ui: UiState,
    group: Node,
    keys: Sequence[str],
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
    commit_inspector_draft(scene, ui, group)
    if next_key is None:
        ui.clear_focus()
        return
    focus_param_key(scene, ui, group.id, next_key)


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
        ui.clear_focus()

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
    handle_inspector_typing(scene, ui, group, [row.key for row in rows])

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    buttons: list[Button] = []
    row: ParamRowRects
    for row in rows:
        field: ParamField = field_for(spec, row.key)
        key: str = row.key

        def step_minus(k: str = key) -> None:
            ui.clear_focus()
            step_generator_param(scene, group.id, k, -1)

        def step_plus(k: str = key) -> None:
            ui.clear_focus()
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
                select_all=edit.select_all,
            )
            continue
        draw_text_field(
            font,
            row.value,
            format_param(
                field,
                params_with_defaults(generator.kind, generator.params)[row.key],
            ),
            focused=False,
            center_unfocused=True,
        )

    button: Button
    for button in buttons:
        draw_button(button, font)
