"""inspector for road sections: import, selection, and polyline edits"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    Vector3,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_pressed,
)

from builder.commands.command_context import SceneContext
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.road_sections.edit import (
    bridge_endpoints,
    delete_polylines,
    insert_vertex_between,
    merge_polylines,
    move_polylines_between_groups,
    normalize_usage_name,
    polyline_ids_shorter_than,
    set_usage_group_on_node,
    simplify_polylines,
    split_polyline_at_vertex,
)
from builder.heightfield.road_sections.fit import curve_merge_span_m
from builder.heightfield.road_sections.fit_report import (
    flagged_rows,
    kit_for_group,
    report_usage_group_polylines,
    write_fit_report,
)
from builder.heightfield.road_sections.part_catalog import (
    RoadPartsCatalog,
    require_kit_for_usage,
    usage_to_kit,
)
from builder.heightfield.road_sections.place import (
    drop_placements,
    place_polylines,
    regenerate_road_piece_children,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
    polyline_by_id,
)
from builder.heightfield.road_sections.shp_import import simplify_eps_m
from builder.heightfield.vector.coords import ordered_child_nodes
from builder.scene.ids import new_node_id
from builder.scene.scene import Scene
from builder.scene.scene_types import Node, transform_at
from builder.session_journal import SessionJournal
from builder.ui.file_browser import (
    FileBrowserPurpose,
    make_road_shp_browser,
    make_tb_roads_export_browser,
    remembered_browser_directory,
)
from builder.ui.list_picker import (
    ListPickerPurpose,
    list_picker_new_item,
    make_list_picker,
)
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
    is_point_in_rect,
    update_buttons,
)

road_sections_panel: str = "road_sections"
shorter_than_key: str = "shorter_than"
move_usage_name_key: str = "move_usage_name"
road_select_shorter_than_min_m: float = 0.1
road_select_shorter_than_step_m: float = 0.1


@dataclass
class RoadFloatRow:
    """± / type-in for select-shorter-than threshold"""

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle


@dataclass
class RoadSectionsInspectorLayout:
    """buttons and optional float row from update"""

    buttons: list[Button] = field(default_factory=list)
    float_row: RoadFloatRow | None = None
    move_name_rect: Rectangle | None = None
    project_checkbox: Rectangle | None = None
    project_onto_shadow: bool = False
    flatten_checkbox: Rectangle | None = None
    ribbon_flatten: bool = False


def format_road_float(value: float) -> str:
    """format threshold metres for the type-in field"""
    return f"{float(value):.4g}"


def parse_road_float(text: str) -> float | None:
    """parse typed threshold text"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def clamp_shorter_than_m(value: float) -> float:
    """clamp select-shorter-than threshold"""
    return max(road_select_shorter_than_min_m, float(value))


def selected_road_sections_layer(scene: Scene) -> Node | None:
    """sole selected road sections layer, or layer owning selected shp/usage"""
    if len(scene.selected_ids) != 1:
        return None
    node: Node | None = scene.nodes.get(scene.selected_ids[0])
    if node is None:
        return None
    current: Node | None = node
    while current is not None:
        if current.road_sections_layer is not None:
            return current
        if current.parent_id is None:
            return None
        current = scene.nodes.get(current.parent_id)
    return None


def selected_usage_group_node(scene: Scene) -> Node | None:
    """sole selected usage group, or none when selection is empty/multi/non-usage"""
    if len(scene.selected_ids) != 1:
        return None
    selected: Node | None = scene.nodes.get(scene.selected_ids[0])
    if selected is None or selected.road_usage_group is None:
        return None
    return selected


def replace_polyline_selection(
    scene: Scene,
    ui: UiState,
    usage_node: Node,
    polyline_ids: list[str],
) -> None:
    """replace compact polyline selection within a usage group"""
    scene.set_selection([usage_node.id])
    ui.road_polyline_usage_group_id = usage_node.id
    ui.road_polyline_ids = list(polyline_ids)
    ui.road_endpoints.clear()
    count: int = len(polyline_ids)
    ui.status = f"Selected {count} polyline" + ("" if count == 1 else "s")


def apply_road_kit_choice(
    scene: Scene,
    ui: UiState,
    usage_node_id: str,
    kit: str,
    journal: SessionJournal | None,
    *,
    road_parts: RoadPartsCatalog | None = None,
    scene_context: SceneContext | None = None,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> None:
    """set usage-group kit from list picker confirm; clears placements"""
    usage_node: Node | None = scene.nodes.get(usage_node_id)
    if usage_node is None or usage_node.road_usage_group is None:
        ui.status = "Kit target missing"
        return
    apply_usage_group_edit(
        scene,
        ui,
        usage_node,
        replace(
            usage_node.road_usage_group,
            kit=kit,
            placements=(),
        ),
        journal,
        "kit",
        road_parts=road_parts,
        scene_context=scene_context,
        heightmap_catalog=heightmap_catalog,
    )


def sibling_usage_names(scene: Scene, usage_node: Node) -> list[str]:
    """usage labels under the same shapefile parent, excluding this node"""
    if usage_node.parent_id is None:
        return []
    names: list[str] = []
    sibling: Node
    for sibling in ordered_child_nodes(scene.nodes, usage_node.parent_id):
        if sibling.id == usage_node.id or sibling.road_usage_group is None:
            continue
        names.append(sibling.road_usage_group.usage)
    return names


def move_usage_picker_items(scene: Scene, usage_node: Node) -> list[str]:
    """dest labels for Move to…: siblings, known classes, then New…"""
    current: str = ""
    if usage_node.road_usage_group is not None:
        current = usage_node.road_usage_group.usage.strip().lower()
    names: set[str] = set()
    name: str
    for name in sibling_usage_names(scene, usage_node):
        key: str = name.strip().lower()
        if key and key != current:
            names.add(key)
    for name in usage_to_kit.keys():
        key = name.strip().lower()
        if key and key != current:
            names.add(key)
    items: list[str] = sorted(names)
    items.append(list_picker_new_item)
    return items


def begin_move_usage_name(ui: UiState, source_usage_id: str) -> None:
    """enter custom usage-name typing after New…"""
    ui.clear_focus()
    ui.road_move_name_source_id = source_usage_id
    ui.focus = FieldId(panel=road_sections_panel, key=move_usage_name_key)
    ui.edit = begin_edit("")
    ui.status = "Name new usage group…"


def apply_move_polylines_to_usage(
    scene: Scene,
    ui: UiState,
    source_usage_id: str,
    dest_usage_name: str,
    journal: SessionJournal | None,
    *,
    road_parts: RoadPartsCatalog | None = None,
    scene_context: SceneContext | None = None,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> None:
    """move selected polylines into an existing or new sibling usage group"""
    try:
        dest_name: str = normalize_usage_name(dest_usage_name)
    except ValueError as exc:
        ui.status = str(exc)
        return
    source_node: Node | None = scene.nodes.get(source_usage_id)
    if source_node is None or source_node.road_usage_group is None:
        ui.status = "Move source missing"
        return
    if source_node.parent_id is None:
        ui.status = "Usage group has no shapefile parent"
        return
    shp_id: str = source_node.parent_id
    shp_node: Node | None = scene.nodes.get(shp_id)
    if shp_node is None or shp_node.road_shp is None:
        ui.status = "Shapefile parent missing"
        return
    polyline_ids: list[str] = list(ui.road_polyline_ids)
    if not polyline_ids:
        ui.status = "No polylines selected"
        return
    if (
        ui.road_polyline_usage_group_id is not None
        and ui.road_polyline_usage_group_id != source_usage_id
    ):
        ui.status = "Selection is not on the move source"
        return

    dest_node: Node | None = None
    sibling: Node
    for sibling in ordered_child_nodes(scene.nodes, shp_id):
        group: RoadUsageGroupRecipe | None = sibling.road_usage_group
        if group is None:
            continue
        if group.usage.strip().lower() == dest_name:
            dest_node = sibling
            break
    if dest_node is not None and dest_node.id == source_usage_id:
        ui.status = "Already in that usage group"
        return
    created: bool = False
    if dest_node is None:
        dest_node = Node(
            id=new_node_id(),
            parent_id=shp_id,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_usage_group=RoadUsageGroupRecipe(
                usage=dest_name,
                kit=require_kit_for_usage(dest_name),
            ),
            name=dest_name,
        )
        created = True

    assert dest_node.road_usage_group is not None
    source_group: RoadUsageGroupRecipe
    dest_group: RoadUsageGroupRecipe
    source_group, dest_group = move_polylines_between_groups(
        source_node.road_usage_group,
        dest_node.road_usage_group,
        polyline_ids,
    )
    moved_count: int = len(source_node.road_usage_group.polylines) - len(
        source_group.polylines
    )
    if moved_count <= 0:
        ui.status = "No matching polylines to move"
        return

    updated_source: Node = set_usage_group_on_node(source_node, source_group)
    updated_dest: Node = (
        replace(dest_node, road_usage_group=dest_group)
        if created
        else set_usage_group_on_node(dest_node, dest_group)
    )
    scene.add_nodes([updated_source, updated_dest])
    if road_parts is not None and scene_context is not None:
        regenerate_road_piece_children(
            scene,
            updated_source.id,
            road_parts,
            has_mesh=scene_context.has_mesh,
            register_mesh=scene_context.register_mesh,
            shader=scene_context.mesh_shader(),
            heightmap_catalog=heightmap_catalog,
        )
        regenerate_road_piece_children(
            scene,
            updated_dest.id,
            road_parts,
            has_mesh=scene_context.has_mesh,
            register_mesh=scene_context.register_mesh,
            shader=scene_context.mesh_shader(),
            heightmap_catalog=heightmap_catalog,
        )
    if heightmap_catalog is not None and (
        source_group.ribbon_flatten
        or dest_group.ribbon_flatten
        or created
    ):
        from builder.heightfield.road_sections.rasterize import (
            regenerate_road_sections_touching,
        )

        regenerate_road_sections_touching(
            heightmap_catalog,
            scene.nodes,
            [updated_source.id, updated_dest.id],
            road_parts=road_parts,
        )
    ui.road_move_name_source_id = None
    replace_polyline_selection(
        scene,
        ui,
        updated_dest,
        polyline_ids,
    )
    if journal is not None:
        journal.record(
            "move_polylines",
            source_usage_id=source_usage_id,
            dest_usage_id=updated_dest.id,
            dest_usage=dest_name,
            count=moved_count,
        )
    ui.status = f"Moved {moved_count} line" + (
        "" if moved_count == 1 else "s"
    ) + f" → {dest_name}"


def apply_usage_group_edit(
    scene: Scene,
    ui: UiState,
    usage_node: Node,
    group: RoadUsageGroupRecipe,
    journal: SessionJournal | None,
    action: str,
    *,
    road_parts: RoadPartsCatalog | None = None,
    scene_context: SceneContext | None = None,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> None:
    """write a mutated usage group back into the scene"""
    scene.add_nodes([set_usage_group_on_node(usage_node, group)])
    if road_parts is not None and scene_context is not None:
        regenerate_road_piece_children(
            scene,
            usage_node.id,
            road_parts,
            has_mesh=scene_context.has_mesh,
            register_mesh=scene_context.register_mesh,
            shader=scene_context.mesh_shader(),
            heightmap_catalog=heightmap_catalog,
        )
    # skip full-layer ribbon rebuild on place/edit when flatten is off
    if heightmap_catalog is not None and (
        group.ribbon_flatten or action == "ribbon_flatten"
    ):
        from builder.heightfield.road_sections.rasterize import (
            regenerate_road_sections_touching,
        )

        regenerate_road_sections_touching(
            heightmap_catalog,
            scene.nodes,
            [usage_node.id],
            road_parts=road_parts,
        )
    # drop selection entries that no longer exist
    if ui.road_polyline_usage_group_id == usage_node.id:
        known: set[str] = {polyline.id for polyline in group.polylines}
        ui.road_polyline_ids = [
            polyline_id
            for polyline_id in ui.road_polyline_ids
            if polyline_id in known
        ]
        if not ui.road_polyline_ids:
            ui.clear_road_polyline_selection()
        else:
            ui.road_endpoints = [
                endpoint
                for endpoint in ui.road_endpoints
                if endpoint.polyline_id in known
            ]
    if journal is not None:
        journal.record(action, usage_group_id=usage_node.id)
    ui.status = f"Road {action.replace('_', ' ')}"


def update_road_sections_inspector(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    layer_node: Node,
    *,
    journal: SessionJournal | None = None,
    road_parts: RoadPartsCatalog | None = None,
    scene_context: SceneContext | None = None,
    heightmap_catalog: HeightmapCatalog | None = None,
) -> RoadSectionsInspectorLayout:
    """layout/input: import, selection tools, and polyline edit actions"""
    layout: RoadSectionsInspectorLayout = RoadSectionsInspectorLayout()
    if layer_node.road_sections_layer is None:
        return layout
    buttons: list[Button] = []
    x: float = area.x + ui_pad
    y: float = area.y + ui_pad + float(ui_font_size) + ui_button_gap
    inner_w: float = area.width - 2.0 * ui_pad
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)

    def open_import() -> None:
        ui.clear_focus()
        ui.list_picker = None
        ui.road_sections_import_target = layer_node.id
        ui.file_browser = make_road_shp_browser(
            remembered_browser_directory(
                ui.browser_directories,
                FileBrowserPurpose.import_road_shp,
                None,
            )
        )
        ui.status = "Import road shapefile…"

    def open_export_tb() -> None:
        ui.clear_focus()
        ui.list_picker = None
        ui.road_sections_import_target = layer_node.id
        ui.file_browser = make_tb_roads_export_browser(
            remembered_browser_directory(
                ui.browser_directories,
                FileBrowserPurpose.export_tb_roads,
                None,
            )
        )
        ui.status = "Export TB road objects…"

    buttons.append(
        Button(
            label="Import SHP…",
            on_click=open_import,
            rect=Rectangle(x, y, inner_w * 0.5 - ui_button_gap * 0.5, row_h),
        )
    )
    buttons.append(
        Button(
            label="Export TB…",
            on_click=open_export_tb,
            rect=Rectangle(
                x + inner_w * 0.5 + ui_button_gap * 0.5,
                y,
                inner_w * 0.5 - ui_button_gap * 0.5,
                row_h,
            ),
        )
    )
    y += row_h + ui_button_gap

    usage_node: Node | None = selected_usage_group_node(scene)
    selected_count: int = len(ui.road_polyline_ids)
    can_edit_lines: bool = (
        usage_node is not None
        and usage_node.road_usage_group is not None
        and selected_count > 0
    )
    float_row: RoadFloatRow | None = None
    move_name_rect: Rectangle | None = None

    if ui.focus is not None and ui.focus.panel == road_sections_panel:
        if ui.edit is not None:
            action: TextAction = handle_text_keys(ui.edit)
            focus_key: str = ui.focus.key
            match action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    if focus_key == move_usage_name_key:
                        ui.road_move_name_source_id = None
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    if focus_key == move_usage_name_key:
                        source_id: str | None = ui.road_move_name_source_id
                        typed_name: str = ui.edit.text
                        ui.clear_focus()
                        if source_id is not None:
                            apply_move_polylines_to_usage(
                                scene,
                                ui,
                                source_id,
                                typed_name,
                                journal,
                                road_parts=road_parts,
                                scene_context=scene_context,
                                heightmap_catalog=heightmap_catalog,
                            )
                    else:
                        parsed: float | None = parse_road_float(ui.edit.text)
                        if parsed is not None:
                            ui.road_select_shorter_than_m = clamp_shorter_than_m(
                                parsed
                            )
                        ui.clear_focus()

    if usage_node is not None and usage_node.road_usage_group is not None:
        group: RoadUsageGroupRecipe = usage_node.road_usage_group

        def select_all() -> None:
            ui.clear_focus()
            replace_polyline_selection(
                scene,
                ui,
                usage_node,
                [polyline.id for polyline in group.polylines],
            )

        def select_shorter() -> None:
            ui.clear_focus()
            ids: list[str] = polyline_ids_shorter_than(
                group, ui.road_select_shorter_than_m
            )
            replace_polyline_selection(scene, ui, usage_node, ids)

        buttons.append(
            Button(
                label="Select all",
                on_click=select_all,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

        label_rect: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        controls_y: float = y + float(ui_font_size) + 4.0
        minus_rect: Rectangle = Rectangle(x, controls_y, step_w, row_h)
        plus_rect: Rectangle = Rectangle(
            x + inner_w - step_w, controls_y, step_w, row_h
        )
        value_rect: Rectangle = Rectangle(
            minus_rect.x + minus_rect.width + ui_button_gap,
            controls_y,
            plus_rect.x - (minus_rect.x + minus_rect.width + ui_button_gap * 2),
            row_h,
        )
        float_row = RoadFloatRow(
            key=shorter_than_key,
            label=label_rect,
            minus=minus_rect,
            value=value_rect,
            plus=plus_rect,
        )

        def nudge_minus() -> None:
            ui.clear_focus()
            ui.road_select_shorter_than_m = clamp_shorter_than_m(
                ui.road_select_shorter_than_m - road_select_shorter_than_step_m
            )

        def nudge_plus() -> None:
            ui.clear_focus()
            ui.road_select_shorter_than_m = (
                ui.road_select_shorter_than_m + road_select_shorter_than_step_m
            )

        buttons.append(Button(label="-", on_click=nudge_minus, rect=minus_rect))
        buttons.append(Button(label="+", on_click=nudge_plus, rect=plus_rect))
        if clicked and is_point_in_rect(mouse.x, mouse.y, value_rect):
            ui.focus = FieldId(panel=road_sections_panel, key=shorter_than_key)
            ui.edit = begin_edit(format_road_float(ui.road_select_shorter_than_m))
        y = controls_y + row_h + ui_button_gap

        buttons.append(
            Button(
                label="Select shorter than",
                on_click=select_shorter,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    def apply_edit(updated: RoadUsageGroupRecipe, action: str) -> None:
        if usage_node is None:
            return
        apply_usage_group_edit(
            scene,
            ui,
            usage_node,
            updated,
            journal,
            action,
            road_parts=road_parts,
            scene_context=scene_context,
            heightmap_catalog=heightmap_catalog,
        )

    def delete_selected() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        try:
            updated: RoadUsageGroupRecipe = delete_polylines(
                usage_node.road_usage_group, ui.road_polyline_ids
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "delete_polylines")

    def simplify_selected() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        try:
            updated = simplify_polylines(
                usage_node.road_usage_group,
                ui.road_polyline_ids,
                simplify_eps_m,
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "simplify_polylines")

    def merge_selected() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        try:
            updated, result_ids = merge_polylines(
                usage_node.road_usage_group, ui.road_polyline_ids
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "merge_polylines")
        ui.road_polyline_usage_group_id = usage_node.id
        ui.road_polyline_ids = list(result_ids)
        ui.road_endpoints.clear()
        ui.status = f"Merged selection into {len(result_ids)} polyline" + (
            "" if len(result_ids) == 1 else "s"
        )

    def connect_ends() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        if len(ui.road_endpoints) != 2:
            ui.status = "Select two endpoints to connect"
            return
        try:
            updated = bridge_endpoints(
                usage_node.road_usage_group,
                ui.road_endpoints[0],
                ui.road_endpoints[1],
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        bridge_id: str = updated.polylines[-1].id
        apply_edit(updated, "bridge_endpoints")
        ui.road_polyline_usage_group_id = usage_node.id
        ui.road_polyline_ids = [bridge_id]
        ui.road_endpoints.clear()

    def split_at_point() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        if len(ui.road_endpoints) != 1:
            ui.status = "Select one interior vertex to split"
            return
        endpoint: RoadEndpointRef = ui.road_endpoints[0]
        try:
            updated = split_polyline_at_vertex(
                usage_node.road_usage_group,
                endpoint.polyline_id,
                endpoint.vertex_index,
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "split_polyline")
        ui.road_polyline_usage_group_id = usage_node.id
        # left keeps original id; right is the new neighbor after it
        right_id: str | None = None
        polyline: RoadPolyline
        for index, polyline in enumerate(updated.polylines):
            if polyline.id != endpoint.polyline_id:
                continue
            if index + 1 < len(updated.polylines):
                right_id = updated.polylines[index + 1].id
            break
        ui.road_polyline_ids = (
            [endpoint.polyline_id, right_id]
            if right_id is not None
            else [endpoint.polyline_id]
        )
        ui.road_endpoints.clear()

    def insert_point() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        if len(ui.road_endpoints) != 2:
            ui.status = "Select two vertices on the same polyline"
            return
        try:
            updated, new_ref = insert_vertex_between(
                usage_node.road_usage_group,
                ui.road_endpoints[0],
                ui.road_endpoints[1],
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "insert_vertex")
        ui.road_polyline_usage_group_id = usage_node.id
        ui.road_polyline_ids = [new_ref.polyline_id]
        ui.road_endpoints = [new_ref]

    def place_selected() -> None:
        if (
            usage_node is None
            or usage_node.road_usage_group is None
            or road_parts is None
        ):
            return
        try:
            updated = place_polylines(
                usage_node.road_usage_group, ui.road_polyline_ids, road_parts
            )
        except ValueError as exc:
            ui.status = str(exc)
            return
        apply_edit(updated, "place_road_pieces")
        selected: set[str] = set(ui.road_polyline_ids)
        placed_count: int = sum(
            1
            for placement in updated.placements
            if placement.polyline_id in selected
        )
        ui.status = f"Placed {placed_count} road piece" + (
            "" if placed_count == 1 else "s"
        )

    def clear_selected_pieces() -> None:
        if usage_node is None or usage_node.road_usage_group is None:
            return
        updated = drop_placements(
            usage_node.road_usage_group, ui.road_polyline_ids
        )
        apply_edit(updated, "clear_road_pieces")

    def report_selected_fit() -> None:
        if (
            usage_node is None
            or usage_node.road_usage_group is None
            or road_parts is None
        ):
            return
        road_parts.ensure_loaded()
        try:
            kit = kit_for_group(usage_node.road_usage_group, road_parts.kits)
        except ValueError as exc:
            ui.status = str(exc)
            return
        rows = report_usage_group_polylines(
            usage_node.road_usage_group, ui.road_polyline_ids, kit
        )
        if not rows:
            ui.status = "Fit report: no placements on selection"
            return
        report_path = write_fit_report(
            rows, merge_span=curve_merge_span_m(kit)
        )
        bad_count: int = len(flagged_rows(rows))
        ui.status = (
            f"Fit report: {bad_count} flagged of {len(rows)} → {report_path}"
        )

    if can_edit_lines:
        buttons.append(
            Button(
                label=f"Delete line ({selected_count})",
                on_click=delete_selected,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

        def open_move_picker() -> None:
            if usage_node is None or usage_node.road_usage_group is None:
                return
            if ui.file_browser is not None:
                return
            items: list[str] = move_usage_picker_items(scene, usage_node)
            ui.clear_focus()
            ui.road_move_name_source_id = None
            ui.list_picker = make_list_picker(
                purpose=ListPickerPurpose.road_move_usage,
                title="Move to usage",
                items=items,
                selected=0,
                target_id=usage_node.id,
            )
            ui.status = "Move to usage…"

        buttons.append(
            Button(
                label="Move to…",
                on_click=open_move_picker,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        buttons.append(
            Button(
                label="Simplify",
                on_click=simplify_selected,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        if road_parts is not None:
            buttons.append(
                Button(
                    label="Place pieces",
                    on_click=place_selected,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
            y += row_h + ui_button_gap
            buttons.append(
                Button(
                    label="Clear pieces",
                    on_click=clear_selected_pieces,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
            y += row_h + ui_button_gap
            buttons.append(
                Button(
                    label="Fit report",
                    on_click=report_selected_fit,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
            y += row_h + ui_button_gap

    if ui.road_move_name_source_id is not None:
        move_name_rect = Rectangle(x, y, inner_w, row_h)

        def confirm_named_move() -> None:
            source_id_confirm: str | None = ui.road_move_name_source_id
            if source_id_confirm is None:
                return
            typed: str = ui.edit.text if ui.edit is not None else ""
            ui.clear_focus()
            apply_move_polylines_to_usage(
                scene,
                ui,
                source_id_confirm,
                typed,
                journal,
                road_parts=road_parts,
                scene_context=scene_context,
                heightmap_catalog=heightmap_catalog,
            )

        if clicked and is_point_in_rect(mouse.x, mouse.y, move_name_rect):
            already: bool = (
                ui.focus is not None
                and ui.focus.panel == road_sections_panel
                and ui.focus.key == move_usage_name_key
                and ui.edit is not None
            )
            if not already:
                ui.focus = FieldId(
                    panel=road_sections_panel, key=move_usage_name_key
                )
                ui.edit = begin_edit("")
        y += row_h + ui_button_gap
        buttons.append(
            Button(
                label="Confirm move",
                on_click=confirm_named_move,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    project_checkbox: Rectangle | None = None
    project_onto_shadow: bool = False
    flatten_checkbox: Rectangle | None = None
    ribbon_flatten: bool = False
    if (
        usage_node is not None
        and usage_node.road_usage_group is not None
        and road_parts is not None
    ):
        road_parts.ensure_loaded()
        kit_label: str = f"Kit: {usage_node.road_usage_group.kit}"

        def open_kit_picker() -> None:
            if usage_node is None or usage_node.road_usage_group is None:
                return
            if road_parts is None:
                return
            if ui.file_browser is not None:
                return
            names: list[str] = sorted(road_parts.kits.keys())
            if not names:
                return
            current: str = usage_node.road_usage_group.kit
            index: int
            try:
                index = names.index(current)
            except ValueError:
                index = 0
            ui.clear_focus()
            ui.list_picker = make_list_picker(
                purpose=ListPickerPurpose.road_kit,
                title="Select kit",
                items=names,
                selected=index,
                target_id=usage_node.id,
            )
            ui.status = "Select kit…"

        buttons.append(
            Button(
                label=kit_label,
                on_click=open_kit_picker,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
        project_onto_shadow = usage_node.road_usage_group.project_onto_shadow
        project_checkbox = Rectangle(x, y, inner_w, row_h)
        y += row_h + ui_button_gap
        ribbon_flatten = usage_node.road_usage_group.ribbon_flatten
        flatten_checkbox = Rectangle(x, y, inner_w, row_h)
        y += row_h + ui_button_gap
    if can_edit_lines and selected_count >= 2:
        buttons.append(
            Button(
                label="Merge",
                on_click=merge_selected,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
    can_split_at_point: bool = False
    if (
        can_edit_lines
        and usage_node is not None
        and usage_node.road_usage_group is not None
        and len(ui.road_endpoints) == 1
    ):
        split_ref: RoadEndpointRef = ui.road_endpoints[0]
        split_line: RoadPolyline | None = polyline_by_id(
            usage_node.road_usage_group, split_ref.polyline_id
        )
        if (
            split_line is not None
            and split_ref.vertex_index > 0
            and split_ref.vertex_index < len(split_line.points) - 1
        ):
            can_split_at_point = True
    if can_split_at_point:
        buttons.append(
            Button(
                label="Split at point",
                on_click=split_at_point,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
    can_insert_point: bool = (
        can_edit_lines
        and len(ui.road_endpoints) == 2
        and ui.road_endpoints[0].polyline_id == ui.road_endpoints[1].polyline_id
        and ui.road_endpoints[0].vertex_index
        != ui.road_endpoints[1].vertex_index
    )
    if can_insert_point:
        buttons.append(
            Button(
                label="Insert point",
                on_click=insert_point,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap
    can_connect_ends: bool = (
        usage_node is not None
        and usage_node.road_usage_group is not None
        and len(ui.road_endpoints) == 2
        and ui.road_endpoints[0].polyline_id != ui.road_endpoints[1].polyline_id
    )
    if can_connect_ends:
        buttons.append(
            Button(
                label="Connect ends",
                on_click=connect_ends,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    update_buttons(buttons, area)
    if clicked and is_point_in_rect(mouse.x, mouse.y, area):
        on_project: bool = (
            project_checkbox is not None
            and is_point_in_rect(mouse.x, mouse.y, project_checkbox)
        )
        on_flatten: bool = (
            flatten_checkbox is not None
            and is_point_in_rect(mouse.x, mouse.y, flatten_checkbox)
        )
        on_control: bool = any(
            is_point_in_rect(mouse.x, mouse.y, button.rect) for button in buttons
        )
        on_value: bool = (
            float_row is not None
            and is_point_in_rect(mouse.x, mouse.y, float_row.value)
        )
        on_move_name: bool = (
            move_name_rect is not None
            and is_point_in_rect(mouse.x, mouse.y, move_name_rect)
        )
        if on_project and usage_node is not None and usage_node.road_usage_group is not None:
            ui.clear_focus()
            project_onto_shadow = (
                not usage_node.road_usage_group.project_onto_shadow
            )
            apply_edit(
                replace(
                    usage_node.road_usage_group,
                    project_onto_shadow=project_onto_shadow,
                ),
                "project_onto_shadow",
            )
        elif on_flatten and usage_node is not None and usage_node.road_usage_group is not None:
            ui.clear_focus()
            ribbon_flatten = not usage_node.road_usage_group.ribbon_flatten
            apply_edit(
                replace(
                    usage_node.road_usage_group,
                    ribbon_flatten=ribbon_flatten,
                ),
                "ribbon_flatten",
            )
        elif not on_control and not on_value and not on_move_name:
            if ui.focus is not None and ui.focus.panel == road_sections_panel:
                if ui.focus.key == move_usage_name_key:
                    ui.road_move_name_source_id = None
                elif ui.edit is not None:
                    parsed_click: float | None = parse_road_float(ui.edit.text)
                    if parsed_click is not None:
                        ui.road_select_shorter_than_m = clamp_shorter_than_m(
                            parsed_click
                        )
                ui.clear_focus()
    layout.buttons = buttons
    layout.float_row = float_row
    layout.move_name_rect = move_name_rect
    layout.project_checkbox = project_checkbox
    layout.project_onto_shadow = project_onto_shadow
    layout.flatten_checkbox = flatten_checkbox
    layout.ribbon_flatten = ribbon_flatten
    return layout


def draw_road_sections_inspector(
    font: Font,
    area: Rectangle,
    layer_node: Node,
    ui: UiState,
    layout: RoadSectionsInspectorLayout,
) -> None:
    """draw road sections import, selection, and edit controls"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(area, 1.0, ui_color_border)
    title: str = layer_node.name if layer_node.name else "Road Sections"
    draw_text_ex(
        font,
        title,
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        1.0,
        ui_color_text,
    )
    if layout.float_row is not None:
        float_row: RoadFloatRow = layout.float_row
        draw_text_ex(
            font,
            "Shorter than (m)",
            Vector2(float_row.label.x, float_row.label.y),
            float(ui_font_size),
            1.0,
            ui_color_text,
        )
        edit: TextEdit | None = ui.focused_edit(
            road_sections_panel, float_row.key
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
                format_road_float(ui.road_select_shorter_than_m),
                focused=False,
            )
    if layout.move_name_rect is not None:
        name_edit: TextEdit | None = ui.focused_edit(
            road_sections_panel, move_usage_name_key
        )
        if name_edit is not None:
            draw_text_field(
                font,
                layout.move_name_rect,
                name_edit.text,
                focused=True,
                caret=name_edit.caret,
                mark=name_edit.mark,
            )
        else:
            draw_text_field(
                font,
                layout.move_name_rect,
                "new usage name",
                focused=False,
            )
    if layout.project_checkbox is not None:
        draw_checkbox_with_label(
            font,
            layout.project_checkbox,
            "Project onto shadow",
            checked=layout.project_onto_shadow,
        )
    if layout.flatten_checkbox is not None:
        draw_checkbox_with_label(
            font,
            layout.flatten_checkbox,
            "Ribbon flatten",
            checked=layout.ribbon_flatten,
        )
    button: Button
    for button in layout.buttons:
        draw_button(button, font)
