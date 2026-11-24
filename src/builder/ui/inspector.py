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
from builder.heightfield.heightfield_types import HeightEditLayer, Heightfield
from builder.heightfield.heightmap_catalog import HeightmapCatalog, HeightmapId
from builder.heightfield.regenerate import HeightfieldSceneHost, set_heightfield
from builder.session_journal import SessionJournal, log_exception
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
from builder.ui.file_browser import (
    FileBrowserPurpose,
    make_heightfield_albedo_browser,
    make_heightmap_browser,
    remembered_browser_directory,
)
from pathlib import Path
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
from builder.ui.ui_state import HeightfieldSizeDraft, UiState
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
    journal: SessionJournal | None = None,
    *,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> None:
    """write one inspector control into params and regenerate"""
    if journal is not None:
        journal.record(
            "inspector_edit",
            group_id=group_id,
            control=control_key(control),
            value=value,
            recipe="generator",
        )
    set_generator_params(
        scene,
        group_id,
        write_control(generator.params, control, value),
        heightmap_catalog=heightmap_catalog,
    )


def apply_heightfield_control_edit(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
    control: ParamControl,
    value: Any,
) -> None:
    """write one inspector control into the heightfield recipe and regenerate"""
    set_heightfield(
        host,
        catalog,
        group_id,
        write_control(heightfield, control, value),
    )


def try_apply_heightfield_control_edit(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    control: ParamControl,
    value: Any,
    journal: SessionJournal | None = None,
) -> None:
    """apply a heightfield edit; surface layout errors in the status line"""
    if journal is not None:
        journal.record(
            "inspector_edit",
            group_id=group_id,
            control=control_key(control),
            value=value,
            recipe="heightfield",
        )
    try:
        apply_heightfield_control_edit(
            host, catalog, group_id, heightfield, control, value
        )
    except ValueError as exc:
        log_exception("heightfield edit failed", exc)
        ui.status = str(exc)


def sync_inspector_focus(ui: UiState, group: Node | None) -> None:
    """clear draft editing when the inspected group changes"""
    if (
        ui.heightfield_size_draft is not None
        and (group is None or ui.heightfield_size_draft.group_id != group.id)
    ):
        ui.heightfield_size_draft = None
    if ui.focus is None or ui.focus.panel != inspector_panel:
        return
    if group is None or ui.focus.owner != group.id:
        ui.clear_focus()


def is_heightfield_size_control(control: ParamControl) -> bool:
    """true for size_x / size_z (deferred until Apply size)"""
    match control:
        case FloatControl(name="size_x") | FloatControl(name="size_z"):
            return True
        case _:
            return False


def heightfield_with_size_draft(
    heightfield: Heightfield,
    draft: HeightfieldSizeDraft | None,
    group_id: str,
) -> Heightfield:
    """overlay staged size values for display and editing"""
    if draft is None or draft.group_id != group_id:
        return heightfield
    return replace(
        heightfield,
        size_x=replace(heightfield.size_x, value=draft.size_x),
        size_z=replace(heightfield.size_z, value=draft.size_z),
    )


def stage_heightfield_size(
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    control: ParamControl,
    value: float,
) -> None:
    """stage a size_x/size_z edit without regenerating tiles"""
    current: Heightfield = heightfield_with_size_draft(
        heightfield, ui.heightfield_size_draft, group_id
    )
    size_x: float = current.size_x.value
    size_z: float = current.size_z.value
    match control:
        case FloatControl(name="size_x"):
            size_x = float(value)
        case FloatControl(name="size_z"):
            size_z = float(value)
        case _:
            raise ValueError("stage_heightfield_size requires size_x or size_z")
    ui.heightfield_size_draft = HeightfieldSizeDraft(
        group_id=group_id,
        size_x=size_x,
        size_z=size_z,
    )


def size_draft_is_dirty(
    heightfield: Heightfield, draft: HeightfieldSizeDraft
) -> bool:
    """true when staged sizes differ from the live recipe"""
    return (
        draft.size_x != heightfield.size_x.value
        or draft.size_z != heightfield.size_z.value
    )


def apply_heightfield_size_draft(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    journal: SessionJournal | None = None,
) -> None:
    """commit staged size_x/size_z and regenerate"""
    draft: HeightfieldSizeDraft | None = ui.heightfield_size_draft
    if draft is None or draft.group_id != group_id:
        return
    if not size_draft_is_dirty(heightfield, draft):
        ui.heightfield_size_draft = None
        return
    edited: Heightfield = replace(
        heightfield,
        size_x=replace(heightfield.size_x, value=draft.size_x),
        size_z=replace(heightfield.size_z, value=draft.size_z),
    )
    if journal is not None:
        journal.record(
            "heightfield_apply_size",
            group_id=group_id,
            size_x=draft.size_x,
            size_z=draft.size_z,
        )
    try:
        set_heightfield(
            host,
            catalog,
            group_id,
            edited,
        )
    except ValueError as exc:
        log_exception("heightfield size apply failed", exc)
        ui.status = str(exc)
        return
    ui.heightfield_size_draft = None


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
    journal: SessionJournal | None = None,
    *,
    heightmap_catalog: HeightmapCatalog | None = None,
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
    apply_control_edit(
        scene,
        group.id,
        generator,
        control,
        parsed,
        journal=journal,
        heightmap_catalog=heightmap_catalog,
    )


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
    journal: SessionJournal | None = None,
    *,
    heightmap_catalog: HeightmapCatalog | None = None,
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
    commit_inspector_draft(
        scene,
        ui,
        group,
        controls,
        journal=journal,
        heightmap_catalog=heightmap_catalog,
    )
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
    heightmap_catalog: HeightmapCatalog | None = None,
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
            heightmap_catalog=heightmap_catalog,
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
                    heightmap_catalog=heightmap_catalog,
                )

            def seed_plus() -> None:
                clear()
                set_generator_pattern(
                    scene,
                    group.id,
                    replace(pattern, seed=pattern.seed + 1),
                    points=points,
                    heightmap_catalog=heightmap_catalog,
                )

            def seed_reroll() -> None:
                clear()
                set_generator_pattern(
                    scene,
                    group.id,
                    replace(pattern, seed=randrange(0, 1_000_000)),
                    points=points,
                    heightmap_catalog=heightmap_catalog,
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
                scene,
                group.id,
                set_slot(pattern, s, active_mesh_id),
                points=points,
                heightmap_catalog=heightmap_catalog,
            )

        def remove(s: int = slot) -> None:
            clear()
            set_generator_pattern(
                scene,
                group.id,
                remove_slot(pattern, s),
                points=points,
                heightmap_catalog=heightmap_catalog,
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
            heightmap_catalog=heightmap_catalog,
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


def set_heightfield_edit_layers(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    edit_layers: tuple[HeightEditLayer, ...],
    journal: SessionJournal | None = None,
) -> None:
    """replace heightfield edit layers and regenerate; surface layout errors"""
    if journal is not None:
        journal.record(
            "heightfield_edit_layers",
            group_id=group_id,
            layer_count=len(edit_layers),
        )
    try:
        set_heightfield(
            host,
            catalog,
            group_id,
            replace(heightfield, edit_layers=edit_layers),
        )
    except ValueError as exc:
        log_exception("heightfield layer edit failed", exc)
        ui.status = str(exc)


def set_heightfield_base(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    base: HeightmapId | None,
    journal: SessionJournal | None = None,
) -> None:
    """replace heightfield base map and regenerate"""
    if journal is not None:
        journal.record(
            "heightfield_base",
            group_id=group_id,
            base="" if base is None else base.name,
        )
    try:
        set_heightfield(
            host,
            catalog,
            group_id,
            replace(heightfield, base=base),
        )
    except ValueError as exc:
        log_exception("heightfield base edit failed", exc)
        ui.status = str(exc)


def set_heightfield_albedo(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    albedo_path: str | None,
    journal: SessionJournal | None = None,
) -> None:
    """set or clear the heightfield albedo image path"""
    if journal is not None:
        journal.record(
            "heightfield_albedo",
            group_id=group_id,
            albedo_path="" if albedo_path is None else albedo_path,
        )
    try:
        set_heightfield(
            host,
            catalog,
            group_id,
            replace(heightfield, albedo_path=albedo_path),
        )
        ui.status = (
            "Cleared heightfield albedo"
            if albedo_path is None
            else f"Albedo: {Path(albedo_path).name}"
        )
    except ValueError as exc:
        log_exception("heightfield albedo edit failed", exc)
        ui.status = str(exc)


def update_heightfield_inspector(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    area: Rectangle,
    group: Node,
    journal: SessionJournal | None = None,
) -> tuple[list[Button], list[ParamRowRects]]:
    """inspector for a heightfield group"""
    heightfield: Heightfield | None = group.heightfield
    if heightfield is None:
        return [], []
    scene: Scene = host.nodes
    display_hf: Heightfield = heightfield_with_size_draft(
        heightfield, ui.heightfield_size_draft, group.id
    )
    controls: tuple[ParamControl, ...] = controls_from_params_type(Heightfield)
    layout: ParamLayout = layout_params(area, controls, display_hf)
    rows: list[ParamRowRects] = list(layout.rows)
    text_keys: list[str] = [
        row.key
        for row in rows
        if not isinstance(control_by_key(controls, row.key), EnumControl)
    ]

    def seed_focus_text(focus_key: str, hf: Heightfield) -> str:
        focus_control: ParamControl = control_by_key(controls, focus_key)
        return format_control(focus_control, read_control(hf, focus_control))

    if ui.focus is not None and ui.focus.panel == inspector_panel:
        if ui.focus.owner != group.id:
            ui.clear_focus()
        elif ui.edit is not None:
            action: TextAction = handle_text_keys(ui.edit)
            match action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    control: ParamControl = control_by_key(controls, ui.focus.key)
                    parsed: Any | None = parse_control(control, ui.edit.text)
                    if parsed is not None:
                        if is_heightfield_size_control(control):
                            stage_heightfield_size(
                                ui, group.id, heightfield, control, float(parsed)
                            )
                        else:
                            try_apply_heightfield_control_edit(
                                host,
                                catalog,
                                ui,
                                group.id,
                                heightfield,
                                control,
                                parsed,
                                journal=journal,
                            )
                    refreshed: Node | None = scene.nodes.get(group.id)
                    if refreshed is not None and refreshed.heightfield is not None:
                        heightfield = refreshed.heightfield
                    display_hf = heightfield_with_size_draft(
                        heightfield, ui.heightfield_size_draft, group.id
                    )
                    next_key: str | None = None
                    if action is TextAction.focus_next:
                        next_key = field_after(text_keys, ui.focus.key, 1)
                    elif action is TextAction.focus_prev:
                        next_key = field_after(text_keys, ui.focus.key, -1)
                    if next_key is None:
                        ui.clear_focus()
                    else:
                        ui.focus = FieldId(
                            panel=inspector_panel,
                            key=next_key,
                            owner=group.id,
                        )
                        ui.edit = begin_edit(seed_focus_text(next_key, display_hf))

    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    buttons: list[Button] = []
    row: ParamRowRects
    for row in rows:
        control = control_by_key(controls, row.key)
        match control:
            case EnumControl() as enum_control:
                if clicked:
                    options: tuple[IntEnum, ...] = enum_members(
                        display_hf, enum_control
                    )
                    option_index: int
                    option_rect: Rectangle
                    for option_index, option_rect in enumerate(row.option_rects):
                        if is_point_in_rect(mouse.x, mouse.y, option_rect):
                            ui.clear_focus()
                            try_apply_heightfield_control_edit(
                                host,
                                catalog,
                                ui,
                                group.id,
                                heightfield,
                                enum_control,
                                options[option_index],
                                journal=journal,
                            )
                            break
            case IntControl() | FloatControl() as numeric:
                step: float = control_step(display_hf, numeric)

                def step_minus(
                    c: ParamControl = numeric,
                    s: float = step,
                    hf: Heightfield = heightfield,
                ) -> None:
                    ui.clear_focus()
                    staged: Heightfield = heightfield_with_size_draft(
                        hf, ui.heightfield_size_draft, group.id
                    )
                    current: Any = read_control(staged, c)
                    next_value: float = float(current) - s
                    if is_heightfield_size_control(c):
                        stage_heightfield_size(ui, group.id, hf, c, next_value)
                        return
                    try_apply_heightfield_control_edit(
                        host,
                        catalog,
                        ui,
                        group.id,
                        hf,
                        c,
                        next_value,
                        journal=journal,
                    )

                def step_plus(
                    c: ParamControl = numeric,
                    s: float = step,
                    hf: Heightfield = heightfield,
                ) -> None:
                    ui.clear_focus()
                    staged = heightfield_with_size_draft(
                        hf, ui.heightfield_size_draft, group.id
                    )
                    current = read_control(staged, c)
                    next_value = float(current) + s
                    if is_heightfield_size_control(c):
                        stage_heightfield_size(ui, group.id, hf, c, next_value)
                        return
                    try_apply_heightfield_control_edit(
                        host,
                        catalog,
                        ui,
                        group.id,
                        hf,
                        c,
                        next_value,
                        journal=journal,
                    )

                buttons.append(
                    Button(label="-", on_click=step_minus, rect=row.minus)
                )
                buttons.append(
                    Button(label="+", on_click=step_plus, rect=row.plus)
                )
                if clicked and is_point_in_rect(mouse.x, mouse.y, row.value):
                    ui.focus = FieldId(
                        panel=inspector_panel,
                        key=control_key(control),
                        owner=group.id,
                    )
                    ui.edit = begin_edit(
                        format_control(control, read_control(display_hf, control))
                    )

    y: float = layout.bottom + ui_pad
    x: float = area.x + ui_pad
    inner_w: float = area.width - ui_pad * 2
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)

    draft: HeightfieldSizeDraft | None = ui.heightfield_size_draft
    if (
        draft is not None
        and draft.group_id == group.id
        and size_draft_is_dirty(heightfield, draft)
    ):

        def apply_size(hf: Heightfield = heightfield) -> None:
            ui.clear_focus()
            apply_heightfield_size_draft(
                host, catalog, ui, group.id, hf, journal=journal
            )

        buttons.append(
            Button(
                label="Apply size",
                on_click=apply_size,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    base_id: HeightmapId | None = heightfield.base
    base_label: str
    if base_id is None:
        base_label = "Base: (none)"
    elif base_id in catalog.entries:
        base_label = f"Base: {catalog.entries[base_id].label}"
    else:
        base_label = f"Base: {base_id.name}"

    def clear_base(hf: Heightfield = heightfield) -> None:
        ui.clear_focus()
        set_heightfield_base(
            host, catalog, ui, group.id, hf, None, journal=journal
        )

    buttons.append(
        Button(
            label=base_label,
            on_click=lambda: None,
            rect=Rectangle(x, y, inner_w - step_w - ui_button_gap, row_h),
        )
    )
    if base_id is not None:
        buttons.append(
            Button(
                label="x",
                on_click=clear_base,
                rect=Rectangle(x + inner_w - step_w, y, step_w, row_h),
            )
        )
    y += row_h + ui_button_gap

    def open_heightmap_browser() -> None:
        ui.clear_focus()
        ui.heightfield_layer_target = group.id
        ui.heightfield_layer_replace_index = None
        ui.file_browser = make_heightmap_browser(
            remembered_browser_directory(
                ui.browser_directories,
                FileBrowserPurpose.import_heightmap,
                None,
            )
        )
        ui.status = (
            "Set base heightmap…"
            if heightfield.base is None
            else "Add edit layer…"
        )

    buttons.append(
        Button(
            label=(
                "+ Base heightmap (.png/.asc)"
                if heightfield.base is None
                else "+ Edit layer (.png/.asc)"
            ),
            on_click=open_heightmap_browser,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    albedo_label: str
    if heightfield.albedo_path is None:
        albedo_label = "Albedo: (none)"
    else:
        albedo_label = f"Albedo: {Path(heightfield.albedo_path).name}"

    def clear_albedo(hf: Heightfield = heightfield) -> None:
        ui.clear_focus()
        set_heightfield_albedo(
            host, catalog, ui, group.id, hf, None, journal=journal
        )

    def open_albedo_browser() -> None:
        ui.clear_focus()
        ui.heightfield_layer_target = group.id
        ui.file_browser = make_heightfield_albedo_browser(
            remembered_browser_directory(
                ui.browser_directories,
                FileBrowserPurpose.import_heightfield_albedo,
                None,
            )
        )
        ui.status = "Set heightfield albedo…"

    buttons.append(
        Button(
            label=albedo_label,
            on_click=open_albedo_browser,
            rect=Rectangle(
                x,
                y,
                inner_w - (step_w + ui_button_gap if heightfield.albedo_path else 0.0),
                row_h,
            ),
        )
    )
    if heightfield.albedo_path is not None:
        buttons.append(
            Button(
                label="x",
                on_click=clear_albedo,
                rect=Rectangle(x + inner_w - step_w, y, step_w, row_h),
            )
        )
    update_buttons(buttons, area)
    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_control: bool = any(
            is_point_in_rect(mouse.x, mouse.y, button.rect) for button in buttons
        )
        on_value: bool = any(
            is_point_in_rect(mouse.x, mouse.y, row.value) for row in rows
        )
        if not on_control and not on_value:
            ui.clear_focus()
    return buttons, rows


def update_inspector(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    group: Node,
    catalog: MeshCatalog,
    active_mesh_id: MeshId,
    scene_context: HeightfieldSceneHost | None = None,
    heightmap_catalog: HeightmapCatalog | None = None,
    journal: SessionJournal | None = None,
) -> tuple[list[Button], list[ParamRowRects]]:
    """handle inspector input; return stepper/bake buttons and row geometry"""
    if group.heightfield is not None:
        if scene_context is None or heightmap_catalog is None:
            return [], []
        return update_heightfield_inspector(
            scene_context,
            heightmap_catalog,
            ui,
            area,
            group,
            journal=journal,
        )
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
    handle_inspector_typing(
        scene,
        ui,
        group,
        text_keys,
        controls,
        journal=journal,
        heightmap_catalog=heightmap_catalog,
    )

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
                        scene,
                        group.id,
                        generator,
                        c,
                        not current,
                        journal=journal,
                        heightmap_catalog=heightmap_catalog,
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
                                journal=journal,
                                heightmap_catalog=heightmap_catalog,
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
                        scene,
                        group.id,
                        generator,
                        c,
                        float(current) - s,
                        journal=journal,
                        heightmap_catalog=heightmap_catalog,
                    )

                def step_plus(
                    c: ParamControl = stepped, s: float = step
                ) -> None:
                    ui.clear_focus()
                    current: Any = read_control(generator.params, c)
                    apply_control_edit(
                        scene,
                        group.id,
                        generator,
                        c,
                        float(current) + s,
                        journal=journal,
                        heightmap_catalog=heightmap_catalog,
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
            heightmap_catalog=heightmap_catalog,
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
                heightmap_catalog=heightmap_catalog,
            )
            buttons.extend(point_buttons)

    if generator.kind == "spline":

        def on_add_point() -> None:
            ui.clear_focus()
            add_spline_point(
                scene, group.id, heightmap_catalog=heightmap_catalog
            )
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
    """draw the right inspector panel for a parametric or heightfield group"""
    title: str
    params: Any
    controls: tuple[ParamControl, ...]
    if group.heightfield is not None:
        title = "Heightfield"
        params = heightfield_with_size_draft(
            group.heightfield, ui.heightfield_size_draft, group.id
        )
        controls = controls_from_params_type(Heightfield)
    elif group.generator is not None:
        generator: Generator = group.generator
        title = get_spec(generator.kind).label
        params = generator.params
        controls = controls_from_params_type(get_spec(generator.kind).params_type)
    else:
        return
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        title,
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    row: ParamRowRects
    for row in rows:
        control: ParamControl = control_by_key(controls, row.key)
        value: Any = read_control(params, control)
        match control:
            case BoolControl():
                draw_checkbox_with_label(
                    font,
                    row.value,
                    control_label(params, control),
                    checked=bool(value),
                )
            case EnumControl() as enum_control:
                draw_text_ex(
                    font,
                    control_label(params, control),
                    Vector2(row.label.x, row.label.y),
                    float(ui_font_size),
                    0,
                    ui_color_text,
                )
                options: tuple[IntEnum, ...] = enum_members(params, enum_control)
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
                    control_label(params, control),
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
