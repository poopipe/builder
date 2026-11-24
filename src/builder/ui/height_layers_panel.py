"""heightfield edit-layer panel: list, blend, texture, uv, amp/opacity"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path

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

from builder.generators.param_types import FloatParam
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    HeightLayerSource,
    Heightfield,
    PaintHeightSource,
    ProceduralHeightSource,
    SculptHeightMode,
    SculptHeightSource,
    RoadSectionsHeightSource,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
    move_edit_layer,
)
from builder.heightfield.heightmap_catalog import HeightmapAsset, HeightmapCatalog
from builder.heightfield.regenerate import HeightfieldSceneHost, set_heightfield
from builder.scene.scene_types import Node
from builder.scene.selection import subtree_ids
from builder.session_journal import SessionJournal, log_exception
from builder.ui.file_browser import (
    make_heightmap_browser,
    make_heightmap_export_browser,
)
from builder.ui.inspector import set_heightfield_edit_layers
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

height_layers_panel: str = "height_layers"


class LayerEditFloat(IntEnum):
    """type-in floats on the selected edit layer"""

    amplitude = 1
    offset = 2
    opacity = 3
    uv_scale_u = 4
    uv_scale_v = 5
    uv_offset_u = 6
    uv_offset_v = 7


class BaseHeightFloat(IntEnum):
    """type-in floats on the base height range"""

    height_min = 1
    height_max = 2


@dataclass
class LayerListRow:
    """one row in the layer list (base or edit)"""

    label: str
    detail: str
    edit_index: int | None
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False
    is_selected: bool = False


@dataclass
class LayerFloatRow:
    """geometry for one ± / type-in float control"""

    field: LayerEditFloat | BaseHeightFloat
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


def layer_float_key(field: LayerEditFloat | BaseHeightFloat) -> str:
    """focus id for a selected-layer or base float"""
    if isinstance(field, BaseHeightFloat):
        return f"base:{field.name}"
    return f"edit:{field.name}"


def parse_layer_float_key(key: str) -> LayerEditFloat | None:
    """decode a selected-layer float focus key"""
    parts: list[str] = key.split(":")
    if len(parts) != 2 or parts[0] != "edit":
        return None
    try:
        return LayerEditFloat[parts[1]]
    except KeyError:
        return None


def parse_base_height_float_key(key: str) -> BaseHeightFloat | None:
    """decode a base height-range float focus key"""
    parts: list[str] = key.split(":")
    if len(parts) != 2 or parts[0] != "base":
        return None
    try:
        return BaseHeightFloat[parts[1]]
    except KeyError:
        return None


def format_layer_float(value: float) -> str:
    """format a layer float for the type-in field"""
    return f"{float(value):.4g}"


def parse_layer_float(text: str) -> float | None:
    """parse typed layer float text, or none if invalid"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def layer_float_label(field: LayerEditFloat | BaseHeightFloat) -> str:
    """panel label for a layer or base float"""
    match field:
        case LayerEditFloat.amplitude:
            return "Amplitude (m)"
        case LayerEditFloat.offset:
            return "Offset (m)"
        case LayerEditFloat.opacity:
            return "Opacity"
        case LayerEditFloat.uv_scale_u:
            return "Scale U"
        case LayerEditFloat.uv_scale_v:
            return "Scale V"
        case LayerEditFloat.uv_offset_u:
            return "Offset U"
        case LayerEditFloat.uv_offset_v:
            return "Offset V"
        case BaseHeightFloat.height_min:
            return "Height Min"
        case BaseHeightFloat.height_max:
            return "Height Max"


def layer_float_value(layer: HeightEditLayer, field: LayerEditFloat) -> float:
    """read one selected-layer float"""
    match field:
        case LayerEditFloat.amplitude:
            return float(layer.amplitude)
        case LayerEditFloat.offset:
            return float(layer.offset)
        case LayerEditFloat.opacity:
            return float(layer.opacity.value)
        case LayerEditFloat.uv_scale_u:
            return float(layer.uv_scale_u)
        case LayerEditFloat.uv_scale_v:
            return float(layer.uv_scale_v)
        case LayerEditFloat.uv_offset_u:
            return float(layer.uv_offset_u)
        case LayerEditFloat.uv_offset_v:
            return float(layer.uv_offset_v)


def layer_float_step(field: LayerEditFloat | BaseHeightFloat) -> float:
    """nudge step for ± buttons"""
    match field:
        case LayerEditFloat.opacity:
            return 0.05
        case (
            LayerEditFloat.uv_scale_u
            | LayerEditFloat.uv_scale_v
            | LayerEditFloat.uv_offset_u
            | LayerEditFloat.uv_offset_v
        ):
            return 0.1
        case (
            LayerEditFloat.amplitude
            | LayerEditFloat.offset
            | BaseHeightFloat.height_min
            | BaseHeightFloat.height_max
        ):
            return 0.25


def base_height_float_value(
    heightfield: Heightfield, field: BaseHeightFloat
) -> float:
    """read one base height-range float"""
    match field:
        case BaseHeightFloat.height_min:
            return float(heightfield.base_height_min.value)
        case BaseHeightFloat.height_max:
            return float(heightfield.base_height_max.value)


def apply_base_height_float(
    heightfield: Heightfield, field: BaseHeightFloat, value: float
) -> Heightfield:
    """return heightfield with one base height-range float replaced"""
    match field:
        case BaseHeightFloat.height_min:
            return replace(
                heightfield,
                base_height_min=FloatParam(value, label="Height Min"),
            )
        case BaseHeightFloat.height_max:
            return replace(
                heightfield,
                base_height_max=FloatParam(value, label="Height Max"),
            )


def apply_layer_float(
    layer: HeightEditLayer, field: LayerEditFloat, value: float
) -> HeightEditLayer:
    """return layer with one float field replaced"""
    match field:
        case LayerEditFloat.amplitude:
            return replace(layer, amplitude=value)
        case LayerEditFloat.offset:
            return replace(layer, offset=value)
        case LayerEditFloat.opacity:
            clamped: float = max(0.0, min(1.0, value))
            return replace(
                layer,
                opacity=FloatParam(
                    clamped, label="Opacity", step=0.05, minimum=0.0, maximum=1.0
                ),
            )
        case LayerEditFloat.uv_scale_u:
            return replace(layer, uv_scale_u=value)
        case LayerEditFloat.uv_scale_v:
            return replace(layer, uv_scale_v=value)
        case LayerEditFloat.uv_offset_u:
            return replace(layer, uv_offset_u=value)
        case LayerEditFloat.uv_offset_v:
            return replace(layer, uv_offset_v=value)


def source_label(catalog: HeightmapCatalog, source: HeightLayerSource) -> str:
    """short label for a layer height source"""
    match source:
        case PaintHeightSource(heightmap_id=heightmap_id):
            asset: HeightmapAsset | None = catalog.entries.get(heightmap_id)
            name: str = heightmap_id.name if asset is None else asset.label
            return f"Paint: {name}"
        case SculptHeightSource(heightmap_id=heightmap_id, mode=mode):
            asset = catalog.entries.get(heightmap_id)
            name = heightmap_id.name if asset is None else asset.label
            return f"Sculpt {mode.name}: {name}"
        case CatalogHeightSource(heightmap_id=heightmap_id):
            asset = catalog.entries.get(heightmap_id)
            if asset is None:
                return heightmap_id.name
            return asset.label
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            return f"{procedural_type.name} ({constant_value:g})"
        case VectorHeightSource(layer_node_id=layer_node_id):
            return f"Vector: {layer_node_id}"
        case VectorSculptHeightSource(layer_node_id=layer_node_id):
            return f"Vector sculpt: {layer_node_id}"
        case VectorRoadHeightSource(layer_node_id=layer_node_id):
            return f"Road: {layer_node_id}"
        case RoadSectionsHeightSource(layer_node_id=layer_node_id):
            return f"Road Sections: {layer_node_id}"


def next_blend_mode(current: HeightBlendMode) -> HeightBlendMode:
    """step forward through HeightBlendMode members"""
    modes: list[HeightBlendMode] = list(HeightBlendMode)
    index: int = modes.index(current)
    return modes[(index + 1) % len(modes)]


def prev_blend_mode(current: HeightBlendMode) -> HeightBlendMode:
    """step backward through HeightBlendMode members"""
    modes: list[HeightBlendMode] = list(HeightBlendMode)
    index: int = modes.index(current)
    return modes[(index - 1) % len(modes)]


def sync_height_layers_selection(ui: UiState, group: Node | None) -> None:
    """clamp selected edit layer to the current heightfield"""
    if group is None or group.heightfield is None:
        ui.height_layers_selected_index = None
        if ui.focus is not None and ui.focus.panel == height_layers_panel:
            ui.clear_focus()
        return
    layer_count: int = len(group.heightfield.edit_layers)
    selected: int | None = ui.height_layers_selected_index
    if selected is None:
        return
    if selected < 0 or selected >= layer_count:
        ui.height_layers_selected_index = None if layer_count == 0 else 0
        if ui.focus is not None and ui.focus.panel == height_layers_panel:
            ui.clear_focus()


def set_selected_layer_float(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    layer_index: int,
    field: LayerEditFloat,
    value: float,
    journal: SessionJournal | None = None,
) -> None:
    """set one float on the selected edit layer and regenerate"""
    if layer_index < 0 or layer_index >= len(heightfield.edit_layers):
        return
    layers: list[HeightEditLayer] = list(heightfield.edit_layers)
    layers[layer_index] = apply_layer_float(layers[layer_index], field, value)
    set_heightfield_edit_layers(
        host, catalog, ui, group_id, heightfield, tuple(layers), journal=journal
    )


def set_base_height_float(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    group_id: str,
    heightfield: Heightfield,
    field: BaseHeightFloat,
    value: float,
    journal: SessionJournal | None = None,
) -> None:
    """set base height min or max and regenerate"""
    if journal is not None:
        journal.record(
            "heightfield_base_height",
            group_id=group_id,
            field=field.name,
            value=value,
        )
    updated: Heightfield = apply_base_height_float(heightfield, field, value)
    try:
        set_heightfield(host, catalog, group_id, updated)
    except ValueError as exc:
        log_exception("heightfield base height edit failed", exc)
        ui.status = str(exc)
        return
    ui.status = f"Base {layer_float_label(field)} = {value:g}"


def update_height_layers_panel(
    host: HeightfieldSceneHost,
    catalog: HeightmapCatalog,
    ui: UiState,
    area: Rectangle,
    group: Node | None,
    journal: SessionJournal | None = None,
) -> tuple[
    list[LayerListRow],
    list[LayerFloatRow],
    list[Button],
    Rectangle | None,
    str | None,
]:
    """layout/input for the height layers panel; empty_message when no content"""
    buttons: list[Button] = []
    list_rows: list[LayerListRow] = []
    float_rows: list[LayerFloatRow] = []
    blend_label: Rectangle | None = None
    sync_height_layers_selection(ui, group)

    if group is None or group.heightfield is None:
        return list_rows, float_rows, buttons, blend_label, "Select a heightfield"
    heightfield: Heightfield = group.heightfield
    group_id: str = group.id

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

    # layer list region (fixed height chunk)
    list_capacity: float = min(160.0, max(80.0, area.height * 0.28))
    list_rect: Rectangle = Rectangle(area.x, y, area.width, list_capacity)
    base_label: str
    if heightfield.base is None:
        base_label = "Base"
        base_detail: str = "(none)"
    elif heightfield.base in catalog.entries:
        base_label = "Base"
        base_detail = catalog.entries[heightfield.base].label
    else:
        base_label = "Base"
        base_detail = heightfield.base.name

    entries: list[tuple[str, str, int | None]] = [
        (base_label, base_detail, None),
    ]
    layer_index: int
    layer: HeightEditLayer
    for layer_index, layer in enumerate(heightfield.edit_layers):
        enabled_mark: str = "" if layer.enabled else " (off)"
        detail: str
        match layer.height:
            case SculptHeightSource(mode=sculpt_mode):
                detail = f"sculpt/{sculpt_mode.name}"
            case VectorHeightSource():
                detail = "vector"
            case VectorSculptHeightSource():
                detail = "vector sculpt"
            case VectorRoadHeightSource():
                detail = "road"
            case RoadSectionsHeightSource():
                detail = "road sections"
            case _:
                detail = layer.blend.name
        entries.append(
            (
                f"{layer.name}{enabled_mark}",
                detail,
                layer_index,
            )
        )

    row_pitch: float = row_h + ui_button_gap
    content_h: float = max(0.0, len(entries) * row_pitch - ui_button_gap)
    max_scroll: float = max(0.0, content_h - list_capacity)
    if is_point_in_rect(mouse.x, mouse.y, list_rect):
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            ui.height_layers_panel_scroll -= wheel * row_pitch
    ui.height_layers_panel_scroll = max(
        0.0, min(max_scroll, ui.height_layers_panel_scroll)
    )

    row_y: float = list_rect.y - ui.height_layers_panel_scroll
    pointer_in_list: bool = is_point_in_rect(mouse.x, mouse.y, list_rect)
    entry_label: str
    entry_detail: str
    entry_edit: int | None
    for entry_label, entry_detail, entry_edit in entries:
        row: LayerListRow = LayerListRow(
            label=entry_label,
            detail=entry_detail,
            edit_index=entry_edit,
            rect=Rectangle(x, row_y, inner_w, row_h),
            is_selected=ui.height_layers_selected_index == entry_edit,
        )
        # base row selected when edit selection is none
        if entry_edit is None:
            row.is_selected = ui.height_layers_selected_index is None
        row.is_hovered = is_point_in_rect(mouse.x, mouse.y, row.rect)
        row.is_pressed = row.is_hovered and left_down
        visible: bool = (
            row.rect.y + row.rect.height > list_rect.y
            and row.rect.y < list_rect.y + list_rect.height
        )
        if pointer_in_list and visible and row.is_hovered and left_released:
            ui.height_layers_selected_index = entry_edit
            ui.clear_focus()
        list_rows.append(row)
        row_y += row_pitch

    y = list_rect.y + list_rect.height + ui_button_gap

    def replace_layers(new_layers: tuple[HeightEditLayer, ...]) -> None:
        set_heightfield_edit_layers(
            host,
            catalog,
            ui,
            group_id,
            heightfield,
            new_layers,
            journal=journal,
        )

    def add_paint_layer() -> None:
        from builder.heightfield.paint import append_paint_edit_layer
        from builder.heightfield.regenerate import set_heightfield

        ui.clear_focus()
        cache_dir: Path = Path.cwd() / "heightmaps"
        updated: Heightfield = append_paint_edit_layer(
            catalog, heightfield, cache_dir=cache_dir
        )
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added paint layer"

    def add_sculpt_layer() -> None:
        from builder.heightfield.regenerate import set_heightfield
        from builder.heightfield.sculpt import append_sculpt_edit_layer

        ui.clear_focus()
        cache_dir = Path.cwd() / "heightmaps"
        updated = append_sculpt_edit_layer(
            catalog, heightfield, cache_dir=cache_dir
        )
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added sculpt layer"

    def add_vector_layer() -> None:
        from builder.heightfield.regenerate import set_heightfield
        from builder.heightfield.vector.edit import append_vector_edit_layer

        ui.clear_focus()
        cache_dir = Path.cwd() / "heightmaps"
        updated, new_nodes = append_vector_edit_layer(
            catalog,
            heightfield,
            group_id,
            cache_dir=cache_dir,
        )
        host.add_nodes(new_nodes)
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added vector layer"

    def add_vector_sculpt_layer() -> None:
        from builder.heightfield.regenerate import set_heightfield
        from builder.heightfield.vector.edit import (
            append_vector_sculpt_edit_layer,
        )

        ui.clear_focus()
        cache_dir = Path.cwd() / "heightmaps"
        updated, new_nodes = append_vector_sculpt_edit_layer(
            catalog,
            heightfield,
            group_id,
            cache_dir=cache_dir,
        )
        host.add_nodes(new_nodes)
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added vector sculpt layer"

    def add_road_layer() -> None:
        from builder.heightfield.regenerate import set_heightfield
        from builder.heightfield.vector.edit import (
            append_vector_road_edit_layer,
        )

        ui.clear_focus()
        cache_dir = Path.cwd() / "heightmaps"
        updated, new_nodes = append_vector_road_edit_layer(
            catalog,
            heightfield,
            group_id,
            cache_dir=cache_dir,
        )
        host.add_nodes(new_nodes)
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added road layer"

    def add_road_sections_layer() -> None:
        from builder.heightfield.regenerate import set_heightfield
        from builder.heightfield.road_sections.edit import (
            append_road_sections_edit_layer,
        )

        ui.clear_focus()
        updated, new_nodes = append_road_sections_edit_layer(
            heightfield,
            group_id,
        )
        host.add_nodes(new_nodes)
        set_heightfield(host, catalog, group_id, updated)
        ui.height_layers_selected_index = len(updated.edit_layers) - 1
        ui.status = "Added road sections layer"

    buttons.append(
        Button(
            label="+ Paint layer",
            on_click=add_paint_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Sculpt layer",
            on_click=add_sculpt_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Vector layer",
            on_click=add_vector_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Vector sculpt",
            on_click=add_vector_sculpt_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Road",
            on_click=add_road_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="+ Road Sections",
            on_click=add_road_sections_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    def export_asc() -> None:
        ui.clear_focus()
        ui.heightfield_layer_target = group_id
        ui.heightfield_layer_replace_index = None
        ui.file_browser = make_heightmap_export_browser(
            Path.cwd(), ".asc", "height.asc"
        )
        ui.status = "Export composited height (.asc)…"

    def export_png() -> None:
        ui.clear_focus()
        ui.heightfield_layer_target = group_id
        ui.heightfield_layer_replace_index = None
        ui.file_browser = make_heightmap_export_browser(
            Path.cwd(), ".png", "height.png"
        )
        ui.status = "Export composited height (.png)…"

    buttons.append(
        Button(
            label="Export .asc…",
            on_click=export_asc,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="Export .png…",
            on_click=export_png,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    selected_index: int | None = ui.height_layers_selected_index
    if selected_index is not None and (
        selected_index < 0 or selected_index >= len(heightfield.edit_layers)
    ):
        update_buttons(buttons, area)
        return list_rows, float_rows, buttons, blend_label, None

    if selected_index is None:
        base_fields: tuple[BaseHeightFloat, ...] = (
            BaseHeightFloat.height_min,
            BaseHeightFloat.height_max,
        )
        base_keys: list[str] = [layer_float_key(f) for f in base_fields]

        if ui.focus is not None and ui.focus.panel == height_layers_panel:
            if ui.focus.owner != group_id:
                ui.clear_focus()
            elif ui.edit is not None:
                base_action: TextAction = handle_text_keys(ui.edit)
                match base_action:
                    case TextAction.editing:
                        pass
                    case TextAction.cancel:
                        ui.clear_focus()
                    case (
                        TextAction.commit
                        | TextAction.focus_next
                        | TextAction.focus_prev
                    ):
                        base_focus: BaseHeightFloat | None = (
                            parse_base_height_float_key(ui.focus.key)
                        )
                        if base_focus is not None:
                            base_parsed: float | None = parse_layer_float(
                                ui.edit.text
                            )
                            if base_parsed is not None:
                                set_base_height_float(
                                    host,
                                    catalog,
                                    ui,
                                    group_id,
                                    heightfield,
                                    base_focus,
                                    base_parsed,
                                    journal=journal,
                                )
                        if base_action is TextAction.commit:
                            ui.clear_focus()
                        elif base_action is TextAction.focus_next:
                            try:
                                bki: int = base_keys.index(ui.focus.key)
                            except ValueError:
                                ui.clear_focus()
                            else:
                                next_base_key: str = base_keys[
                                    (bki + 1) % len(base_keys)
                                ]
                                next_base: BaseHeightFloat | None = (
                                    parse_base_height_float_key(next_base_key)
                                )
                                if next_base is not None:
                                    live_base: Node | None = host.nodes.nodes.get(
                                        group_id
                                    )
                                    live_hf: Heightfield = heightfield
                                    if (
                                        live_base is not None
                                        and live_base.heightfield is not None
                                    ):
                                        live_hf = live_base.heightfield
                                    ui.focus = FieldId(
                                        panel=height_layers_panel,
                                        key=next_base_key,
                                        owner=group_id,
                                    )
                                    ui.edit = begin_edit(
                                        format_layer_float(
                                            base_height_float_value(
                                                live_hf, next_base
                                            )
                                        )
                                    )
                        elif base_action is TextAction.focus_prev:
                            try:
                                bki = base_keys.index(ui.focus.key)
                            except ValueError:
                                ui.clear_focus()
                            else:
                                prev_base_key: str = base_keys[
                                    (bki - 1) % len(base_keys)
                                ]
                                prev_base: BaseHeightFloat | None = (
                                    parse_base_height_float_key(prev_base_key)
                                )
                                if prev_base is not None:
                                    live_base = host.nodes.nodes.get(group_id)
                                    live_hf = heightfield
                                    if (
                                        live_base is not None
                                        and live_base.heightfield is not None
                                    ):
                                        live_hf = live_base.heightfield
                                    ui.focus = FieldId(
                                        panel=height_layers_panel,
                                        key=prev_base_key,
                                        owner=group_id,
                                    )
                                    ui.edit = begin_edit(
                                        format_layer_float(
                                            base_height_float_value(
                                                live_hf, prev_base
                                            )
                                        )
                                    )

        live_base_group: Node | None = host.nodes.nodes.get(group_id)
        if (
            live_base_group is not None
            and live_base_group.heightfield is not None
        ):
            heightfield = live_base_group.heightfield

        base_field: BaseHeightFloat
        for base_field in base_fields:
            base_field_key: str = layer_float_key(base_field)
            base_label_rect: Rectangle = Rectangle(
                x, y, inner_w, float(ui_font_size)
            )
            base_controls_y: float = y + float(ui_font_size) + 4.0
            base_minus: Rectangle = Rectangle(x, base_controls_y, step_w, row_h)
            base_plus: Rectangle = Rectangle(
                x + inner_w - step_w, base_controls_y, step_w, row_h
            )
            base_value: Rectangle = Rectangle(
                base_minus.x + base_minus.width + ui_button_gap,
                base_controls_y,
                base_plus.x
                - (base_minus.x + base_minus.width + ui_button_gap * 2),
                row_h,
            )
            float_rows.append(
                LayerFloatRow(
                    field=base_field,
                    label=base_label_rect,
                    minus=base_minus,
                    value=base_value,
                    plus=base_plus,
                )
            )

            def base_field_minus(
                f: BaseHeightFloat = base_field,
                hf: Heightfield = heightfield,
            ) -> None:
                ui.clear_focus()
                current: float = base_height_float_value(hf, f)
                set_base_height_float(
                    host,
                    catalog,
                    ui,
                    group_id,
                    hf,
                    f,
                    current - layer_float_step(f),
                    journal=journal,
                )

            def base_field_plus(
                f: BaseHeightFloat = base_field,
                hf: Heightfield = heightfield,
            ) -> None:
                ui.clear_focus()
                current = base_height_float_value(hf, f)
                set_base_height_float(
                    host,
                    catalog,
                    ui,
                    group_id,
                    hf,
                    f,
                    current + layer_float_step(f),
                    journal=journal,
                )

            buttons.append(
                Button(label="-", on_click=base_field_minus, rect=base_minus)
            )
            buttons.append(
                Button(label="+", on_click=base_field_plus, rect=base_plus)
            )
            if clicked and is_point_in_rect(mouse.x, mouse.y, base_value):
                ui.focus = FieldId(
                    panel=height_layers_panel,
                    key=base_field_key,
                    owner=group_id,
                )
                ui.edit = begin_edit(
                    format_layer_float(
                        base_height_float_value(heightfield, base_field)
                    )
                )
            y = base_controls_y + row_h + ui_button_gap

        update_buttons(buttons, area)
        if clicked and is_point_in_rect(mouse.x, mouse.y, area):
            on_control: bool = any(
                is_point_in_rect(mouse.x, mouse.y, button.rect)
                for button in buttons
            )
            on_value: bool = any(
                is_point_in_rect(mouse.x, mouse.y, row.value)
                for row in float_rows
            )
            if not on_control and not on_value:
                ui.clear_focus()
        return list_rows, float_rows, buttons, blend_label, None

    selected: HeightEditLayer = heightfield.edit_layers[selected_index]
    is_sculpt: bool = isinstance(selected.height, SculptHeightSource)
    is_vector_sculpt: bool = isinstance(
        selected.height, VectorSculptHeightSource
    )
    is_road: bool = isinstance(selected.height, VectorRoadHeightSource)
    is_road_sections: bool = isinstance(
        selected.height, RoadSectionsHeightSource
    )
    is_paint: bool = isinstance(
        selected.height, (PaintHeightSource, VectorHeightSource)
    )
    float_fields: tuple[LayerEditFloat, ...]

    if is_road_sections:
        tex_label = f"Paths: {source_label(catalog, selected.height)}"
        buttons.append(
            Button(
                label=tex_label,
                on_click=lambda: None,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        float_fields = ()
    elif is_vector_sculpt or is_road:
        tex_label = f"Height: {source_label(catalog, selected.height)}"
        buttons.append(
            Button(
                label=tex_label,
                on_click=lambda: None,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        float_fields = (LayerEditFloat.opacity,)
    elif is_sculpt:
        match selected.height:
            case SculptHeightSource(mode=current_mode):
                pass
            case _:
                current_mode = SculptHeightMode.delta

        def toggle_sculpt_mode() -> None:
            from builder.heightfield.regenerate import set_heightfield
            from builder.heightfield.sculpt import set_sculpt_layer_mode

            ui.clear_focus()
            next_mode: SculptHeightMode = (
                SculptHeightMode.absolute
                if current_mode is SculptHeightMode.delta
                else SculptHeightMode.delta
            )
            cache_dir = Path.cwd() / "heightmaps"
            updated = set_sculpt_layer_mode(
                catalog,
                heightfield,
                selected_index,
                next_mode,
                cache_dir=cache_dir,
            )
            set_heightfield(host, catalog, group_id, updated)
            ui.status = f"Sculpt mode: {next_mode.name}"

        buttons.append(
            Button(
                label=f"Mode: {current_mode.name}",
                on_click=toggle_sculpt_mode,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        tex_label = f"Height: {source_label(catalog, selected.height)}"
        buttons.append(
            Button(
                label=tex_label,
                on_click=lambda: None,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        float_fields = (LayerEditFloat.opacity,)
    else:
        blend_label = Rectangle(x, y, inner_w, float(ui_font_size))
        blend_controls_y: float = y + float(ui_font_size) + 4.0
        blend_minus: Rectangle = Rectangle(x, blend_controls_y, step_w, row_h)
        blend_plus: Rectangle = Rectangle(
            x + inner_w - step_w, blend_controls_y, step_w, row_h
        )
        blend_value: Rectangle = Rectangle(
            blend_minus.x + blend_minus.width + ui_button_gap,
            blend_controls_y,
            blend_plus.x
            - (blend_minus.x + blend_minus.width + ui_button_gap * 2),
            row_h,
        )

        def step_blend_prev() -> None:
            ui.clear_focus()
            layers: list[HeightEditLayer] = list(heightfield.edit_layers)
            layers[selected_index] = replace(
                layers[selected_index], blend=prev_blend_mode(selected.blend)
            )
            replace_layers(tuple(layers))

        def step_blend_next() -> None:
            ui.clear_focus()
            layers = list(heightfield.edit_layers)
            layers[selected_index] = replace(
                layers[selected_index], blend=next_blend_mode(selected.blend)
            )
            replace_layers(tuple(layers))

        buttons.append(
            Button(label="-", on_click=step_blend_prev, rect=blend_minus)
        )
        buttons.append(
            Button(
                label=selected.blend.name,
                on_click=lambda: None,
                rect=blend_value,
            )
        )
        buttons.append(
            Button(label="+", on_click=step_blend_next, rect=blend_plus)
        )
        y = blend_controls_y + row_h + ui_button_gap

        tex_label = f"Height: {source_label(catalog, selected.height)}"
        buttons.append(
            Button(
                label=tex_label,
                on_click=lambda: None,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

        if not is_paint:

            def open_browse() -> None:
                ui.clear_focus()
                ui.heightfield_layer_target = group_id
                ui.heightfield_layer_replace_index = selected_index
                ui.file_browser = make_heightmap_browser(Path.cwd())
                ui.status = "Replace layer heightmap…"

            buttons.append(
                Button(
                    label="Browse…",
                    on_click=open_browse,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
            y += row_h + ui_button_gap

        float_fields = (
            LayerEditFloat.amplitude,
            LayerEditFloat.offset,
            LayerEditFloat.opacity,
            LayerEditFloat.uv_scale_u,
            LayerEditFloat.uv_scale_v,
            LayerEditFloat.uv_offset_u,
            LayerEditFloat.uv_offset_v,
        )
    text_keys: list[str] = [layer_float_key(f) for f in float_fields]

    if ui.focus is not None and ui.focus.panel == height_layers_panel:
        if ui.focus.owner != group_id:
            ui.clear_focus()
        elif ui.edit is not None:
            action: TextAction = handle_text_keys(ui.edit)
            match action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    focus_field: LayerEditFloat | None = parse_layer_float_key(
                        ui.focus.key
                    )
                    if focus_field is not None:
                        parsed: float | None = parse_layer_float(ui.edit.text)
                        if parsed is not None:
                            set_selected_layer_float(
                                host,
                                catalog,
                                ui,
                                group_id,
                                heightfield,
                                selected_index,
                                focus_field,
                                parsed,
                                journal=journal,
                            )
                    if action is TextAction.commit:
                        ui.clear_focus()
                    elif action is TextAction.focus_next:
                        try:
                            ki: int = text_keys.index(ui.focus.key)
                        except ValueError:
                            ui.clear_focus()
                        else:
                            next_key: str = text_keys[(ki + 1) % len(text_keys)]
                            next_field = parse_layer_float_key(next_key)
                            if next_field is not None:
                                refreshed: Node | None = host.nodes.nodes.get(group_id)
                                live: HeightEditLayer = selected
                                if (
                                    refreshed is not None
                                    and refreshed.heightfield is not None
                                    and selected_index
                                    < len(refreshed.heightfield.edit_layers)
                                ):
                                    live = refreshed.heightfield.edit_layers[
                                        selected_index
                                    ]
                                ui.focus = FieldId(
                                    panel=height_layers_panel,
                                    key=next_key,
                                    owner=group_id,
                                )
                                ui.edit = begin_edit(
                                    format_layer_float(
                                        layer_float_value(live, next_field)
                                    )
                                )
                    elif action is TextAction.focus_prev:
                        try:
                            ki = text_keys.index(ui.focus.key)
                        except ValueError:
                            ui.clear_focus()
                        else:
                            prev_key: str = text_keys[(ki - 1) % len(text_keys)]
                            prev_field = parse_layer_float_key(prev_key)
                            if prev_field is not None:
                                refreshed = host.nodes.nodes.get(group_id)
                                live = selected
                                if (
                                    refreshed is not None
                                    and refreshed.heightfield is not None
                                    and selected_index
                                    < len(refreshed.heightfield.edit_layers)
                                ):
                                    live = refreshed.heightfield.edit_layers[
                                        selected_index
                                    ]
                                ui.focus = FieldId(
                                    panel=height_layers_panel,
                                    key=prev_key,
                                    owner=group_id,
                                )
                                ui.edit = begin_edit(
                                    format_layer_float(
                                        layer_float_value(live, prev_field)
                                    )
                                )

    # re-read after possible commit
    live_group: Node | None = host.nodes.nodes.get(group_id)
    if live_group is not None and live_group.heightfield is not None:
        heightfield = live_group.heightfield
        if selected_index < len(heightfield.edit_layers):
            selected = heightfield.edit_layers[selected_index]

    edit_field: LayerEditFloat
    for edit_field in float_fields:
        field_key: str = layer_float_key(edit_field)
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
            LayerFloatRow(
                field=edit_field,
                label=label_rect,
                minus=minus_rect,
                value=value_rect,
                plus=plus_rect,
            )
        )

        def field_minus(
            f: LayerEditFloat = edit_field,
            index: int = selected_index,
            hf: Heightfield = heightfield,
        ) -> None:
            ui.clear_focus()
            current: float = layer_float_value(hf.edit_layers[index], f)
            set_selected_layer_float(
                host,
                catalog,
                ui,
                group_id,
                hf,
                index,
                f,
                current - layer_float_step(f),
                journal=journal,
            )

        def field_plus(
            f: LayerEditFloat = edit_field,
            index: int = selected_index,
            hf: Heightfield = heightfield,
        ) -> None:
            ui.clear_focus()
            current = layer_float_value(hf.edit_layers[index], f)
            set_selected_layer_float(
                host,
                catalog,
                ui,
                group_id,
                hf,
                index,
                f,
                current + layer_float_step(f),
                journal=journal,
            )

        buttons.append(Button(label="-", on_click=field_minus, rect=minus_rect))
        buttons.append(Button(label="+", on_click=field_plus, rect=plus_rect))
        if clicked and is_point_in_rect(mouse.x, mouse.y, value_rect):
            ui.focus = FieldId(
                panel=height_layers_panel, key=field_key, owner=group_id
            )
            ui.edit = begin_edit(
                format_layer_float(layer_float_value(selected, edit_field))
            )
        y = controls_y + row_h + ui_button_gap

    def remove_layer() -> None:
        ui.clear_focus()
        layers = list(heightfield.edit_layers)
        removed: HeightEditLayer = layers[selected_index]
        match removed.height:
            case VectorHeightSource(layer_node_id=layer_node_id):
                host.nodes.remove_nodes(
                    subtree_ids(host.nodes.nodes, [layer_node_id])
                )
            case VectorSculptHeightSource(layer_node_id=layer_node_id):
                host.nodes.remove_nodes(
                    subtree_ids(host.nodes.nodes, [layer_node_id])
                )
            case VectorRoadHeightSource(layer_node_id=layer_node_id):
                host.nodes.remove_nodes(
                    subtree_ids(host.nodes.nodes, [layer_node_id])
                )
            case RoadSectionsHeightSource(layer_node_id=layer_node_id):
                host.nodes.remove_nodes(
                    subtree_ids(host.nodes.nodes, [layer_node_id])
                )
            case _:
                pass
        del layers[selected_index]
        ui.height_layers_selected_index = None if len(layers) == 0 else min(
            selected_index, len(layers) - 1
        )
        replace_layers(tuple(layers))

    def toggle_enabled() -> None:
        ui.clear_focus()
        layers = list(heightfield.edit_layers)
        layers[selected_index] = replace(
            layers[selected_index], enabled=not layers[selected_index].enabled
        )
        replace_layers(tuple(layers))

    def move_selected(delta: int) -> None:
        ui.clear_focus()
        moved: tuple[tuple[HeightEditLayer, ...], int] | None = move_edit_layer(
            heightfield.edit_layers, selected_index, delta
        )
        if moved is None:
            return
        new_layers: tuple[HeightEditLayer, ...]
        new_index: int
        new_layers, new_index = moved
        ui.height_layers_selected_index = new_index
        replace_layers(new_layers)

    buttons.append(
        Button(
            label="Enabled" if selected.enabled else "Disabled",
            on_click=toggle_enabled,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    if selected_index > 0:
        buttons.append(
            Button(
                label="Move up",
                on_click=lambda: move_selected(-1),
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
    if selected_index < len(heightfield.edit_layers) - 1:
        buttons.append(
            Button(
                label="Move down",
                on_click=lambda: move_selected(1),
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
    buttons.append(
        Button(
            label="Remove layer",
            on_click=remove_layer,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )

    update_buttons(buttons, area)
    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_control: bool = any(
            is_point_in_rect(mouse.x, mouse.y, button.rect) for button in buttons
        )
        on_value: bool = any(
            is_point_in_rect(mouse.x, mouse.y, row.value) for row in float_rows
        )
        if not on_control and not on_value:
            ui.clear_focus()
    return list_rows, float_rows, buttons, blend_label, None


def draw_layer_list_row(font: Font, row: LayerListRow) -> None:
    """draw one layer list row"""
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
    label_size: float = float(ui_font_size)
    detail_size: float = float(ui_font_size - 4)
    tx: float = row.rect.x + 8.0
    ty: float = row.rect.y + 4.0
    draw_text_ex(
        font, marker + row.label, Vector2(tx, ty), label_size, 0, ui_color_text
    )
    draw_text_ex(
        font,
        row.detail,
        Vector2(tx + 14.0, ty + label_size - 2.0),
        detail_size,
        0,
        ui_color_text,
    )


def draw_height_layers_panel(
    font: Font,
    area: Rectangle,
    ui: UiState,
    group: Node | None,
    list_rows: list[LayerListRow],
    float_rows: list[LayerFloatRow],
    buttons: list[Button],
    blend_label: Rectangle | None,
    empty_message: str | None,
) -> None:
    """draw the height layers panel"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        "Height Layers",
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    if empty_message is not None:
        draw_text_ex(
            font,
            empty_message,
            Vector2(area.x + ui_pad, area.y + ui_pad + float(ui_font_size + ui_pad)),
            float(ui_font_size),
            0,
            ui_color_text,
        )
        return

    title_h: float = float(ui_font_size + ui_pad)
    list_top: float = area.y + ui_pad + title_h
    list_capacity: float = min(160.0, max(80.0, area.height * 0.28))
    begin_scissor_mode(
        int(area.x), int(list_top), int(area.width), int(list_capacity)
    )
    row: LayerListRow
    for row in list_rows:
        draw_layer_list_row(font, row)
    end_scissor_mode()

    if blend_label is not None:
        draw_text_ex(
            font,
            "Blend",
            Vector2(blend_label.x, blend_label.y),
            float(ui_font_size),
            0,
            ui_color_text,
        )

    button: Button
    for button in buttons:
        draw_button(button, font)

    owner: str = "" if group is None else group.id
    float_row: LayerFloatRow
    for float_row in float_rows:
        draw_text_ex(
            font,
            layer_float_label(float_row.field),
            Vector2(float_row.label.x, float_row.label.y),
            float(ui_font_size),
            0,
            ui_color_text,
        )
        edit: TextEdit | None = ui.focused_edit(
            height_layers_panel, layer_float_key(float_row.field), owner
        )
        value_text: str = ""
        if group is not None and group.heightfield is not None:
            if isinstance(float_row.field, BaseHeightFloat):
                if ui.height_layers_selected_index is None:
                    value_text = format_layer_float(
                        base_height_float_value(
                            group.heightfield, float_row.field
                        )
                    )
            elif (
                ui.height_layers_selected_index is not None
                and 0
                <= ui.height_layers_selected_index
                < len(group.heightfield.edit_layers)
            ):
                value_text = format_layer_float(
                    layer_float_value(
                        group.heightfield.edit_layers[
                            ui.height_layers_selected_index
                        ],
                        float_row.field,
                    )
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
                value_text,
                focused=False,
                center_unfocused=True,
            )
