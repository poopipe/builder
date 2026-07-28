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


@dataclass(frozen=True)
class ParamRowRects:
    """hit targets for one inspector param row

    option_rects holds one hit target per enum option, in field.options order
    """

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle
    is_bool: bool = False
    is_enum: bool = False
    option_rects: tuple[Rectangle, ...] = ()
    modifier_index: int | None = None


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


def layout_field_row(
    field: ParamField, x: float, y: float, inner_w: float
) -> tuple[ParamRowRects, float]:
    """place one param row and return it with the y below it"""
    empty: Rectangle = Rectangle(0.0, 0.0, 0.0, 0.0)
    if field.value_type == "bool":
        hit: Rectangle = Rectangle(x, y, inner_w, float(ui_button_height))
        row: ParamRowRects = ParamRowRects(
            key=field.key,
            label=hit,
            minus=empty,
            value=hit,
            plus=empty,
            is_bool=True,
        )
        return row, y + float(ui_button_height) + ui_pad
    if field.value_type == "enum":
        options: tuple[tuple[int, str], ...] = field.options or ()
        enum_label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        controls_y: float = y + float(ui_font_size) + 4.0
        option_count: int = max(1, len(options))
        option_w: float = (
            inner_w - ui_button_gap * (option_count - 1)
        ) / option_count
        option_rects: tuple[Rectangle, ...] = tuple(
            Rectangle(
                x + (option_w + ui_button_gap) * index,
                controls_y,
                option_w,
                float(ui_button_height),
            )
            for index in range(len(options))
        )
        row = ParamRowRects(
            key=field.key,
            label=enum_label,
            minus=empty,
            value=empty,
            plus=empty,
            is_enum=True,
            option_rects=option_rects,
        )
        return row, controls_y + float(ui_button_height) + ui_pad
    label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
    controls_y = y + float(ui_font_size) + 4.0
    minus: Rectangle = Rectangle(
        x, controls_y, float(ui_stepper_width), float(ui_button_height)
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
    row = ParamRowRects(
        key=field.key, label=label, minus=minus, value=value, plus=plus
    )
    return row, controls_y + float(ui_button_height) + ui_pad


def layout_params(
    area: Rectangle, fields: tuple[ParamField, ...], collapsed: set[str]
) -> ParamLayout:
    """place group headers and the rows of ungrouped or expanded params"""
    headers: list[GroupHeaderRect] = []
    rows: list[ParamRowRects] = []
    y: float = area.y + ui_pad + float(ui_font_size) + ui_pad
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    current_group: str = ""
    group_open: bool = True
    field: ParamField
    for field in fields:
        if field.group != current_group:
            current_group = field.group
            if field.group != "":
                group_open = field.group not in collapsed
                headers.append(
                    GroupHeaderRect(
                        title=field.group,
                        rect=Rectangle(x, y, inner_w, group_header_height),
                        expanded=group_open,
                    )
                )
                y += group_header_height + ui_button_gap
            else:
                group_open = True
        if field.group != "" and not group_open:
            continue
        row: ParamRowRects
        row, y = layout_field_row(field, x, y, inner_w)
        rows.append(row)
    return ParamLayout(tuple(headers), tuple(rows), y)


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


def focus_param_field(ui: UiState, group: Node, field: ParamField) -> None:
    """begin typed editing for one generator param"""
    if field.value_type in ("bool", "enum"):
        return
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
            label=f"Mode: {pattern.mode}",
            on_click=cycle_mode,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    if pattern.mode == "random":

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
    layout: ParamLayout = layout_params(area, spec.fields, ui.inspector_collapsed)
    rows: list[ParamRowRects] = list(layout.rows)
    text_keys: list[str] = [
        row.key for row in rows if not row.is_bool and not row.is_enum
    ]
    handle_inspector_typing(scene, ui, group, text_keys)

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
    params: dict[str, ParamValue] = params_with_defaults(
        generator.kind, generator.params
    )
    row: ParamRowRects
    for row in rows:
        field: ParamField = field_for(spec, row.key)
        key: str = row.key
        if row.is_bool:

            def toggle_bool(k: str = key, current: ParamValue = params[key]) -> None:
                ui.clear_focus()
                set_generator_param(scene, group.id, k, 0 if int(current) else 1)

            if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                toggle_bool()
            continue

        if row.is_enum:
            if clicked:
                options: tuple[tuple[int, str], ...] = field.options or ()
                option_index: int
                option_rect: Rectangle
                for option_index, option_rect in enumerate(row.option_rects):
                    if is_point_in_rect(mouse.x, mouse.y, option_rect):
                        ui.clear_focus()
                        set_generator_param(
                            scene, group.id, key, options[option_index][0]
                        )
                        break
            continue

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
        value: ParamValue = params_with_defaults(generator.kind, generator.params)[
            row.key
        ]
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
