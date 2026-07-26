"""scene outliner: a tree of group nodes with select, collapse, and rename"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from pyray import (
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
    get_time,
    is_mouse_button_pressed,
    measure_text_ex,
)

from builder.generators.registry import get_spec
from builder.meshes.mesh_catalog import MeshAsset, MeshCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.text_field import (
    FieldId,
    TextAction,
    begin_edit,
    ctrl_held,
    draw_text_field,
    handle_text_keys,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_button_hover,
    ui_color_input_focus,
    ui_color_panel,
    ui_color_selection,
    ui_color_text,
    ui_double_click_sec,
    ui_font_size,
    ui_pad,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_buttons

outliner_panel: str = "outliner"
outliner_row_height: float = float(ui_font_size) + 6.0
outliner_indent: float = 14.0
outliner_caret_width: float = 16.0


@dataclass
class OutlinerRow:
    """one group row in the outliner tree"""

    node_id: str
    depth: int
    name: str
    summary: str
    has_children: bool
    collapsed: bool
    rect: Rectangle
    caret_rect: Rectangle
    name_rect: Rectangle
    is_selected: bool
    is_parent_target: bool
    is_hovered: bool = False


def group_display_name(node: Node) -> str:
    """name shown in the outliner, derived from the generator when unnamed"""
    if node.name != "":
        return node.name
    if node.generator is not None:
        return get_spec(node.generator.kind).label
    return "Group"


def mesh_summary(nodes: dict[str, Node], group_id: str, catalog: MeshCatalog) -> str:
    """short description of a group's direct meshed children"""
    labels: list[str] = []
    node: Node
    for node in nodes.values():
        if node.parent_id != group_id or node.mesh_id is None:
            continue
        asset: MeshAsset | None = catalog.entries.get(node.mesh_id)
        labels.append(asset.label if asset is not None else node.mesh_id.name)
    if not labels:
        return ""
    if len(set(labels)) == 1:
        return f"{len(labels)}x {labels[0]}"
    return f"{len(labels)} meshes"


def group_children_map(nodes: dict[str, Node]) -> dict[str | None, list[Node]]:
    """map parent id -> child group nodes, keeping scene insertion order"""
    children: dict[str | None, list[Node]] = {}
    node: Node
    for node in nodes.values():
        if node.mesh_id is not None:
            continue
        children.setdefault(node.parent_id, []).append(node)
    return children


def flatten_groups(
    children: dict[str | None, list[Node]],
    collapsed: set[str],
) -> list[tuple[Node, int, bool]]:
    """depth-first (node, depth, has_children), skipping collapsed subtrees"""
    result: list[tuple[Node, int, bool]] = []

    def walk(parent_id: str | None, depth: int) -> None:
        node: Node
        for node in children.get(parent_id, []):
            has_children: bool = bool(children.get(node.id))
            result.append((node, depth, has_children))
            if has_children and node.id not in collapsed:
                walk(node.id, depth + 1)

    walk(None, 0)
    return result


def commit_outliner_rename(scene: Scene, ui: UiState) -> None:
    """write the in-progress rename to its node, then drop focus"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != outliner_panel:
        return
    node: Node | None = scene.nodes.get(ui.focus.owner)
    if node is not None:
        scene.set_node(replace(node, name=ui.edit.text.strip()))
    ui.clear_focus()


def begin_outliner_rename(ui: UiState, node: Node) -> None:
    """start editing a group's display name"""
    ui.focus = FieldId(panel=outliner_panel, key="name", owner=node.id)
    ui.edit = begin_edit(group_display_name(node))


def handle_outliner_typing(scene: Scene, ui: UiState) -> None:
    """apply keyboard input to the focused rename field"""
    if ui.focus is None or ui.edit is None or ui.focus.panel != outliner_panel:
        return
    if scene.nodes.get(ui.focus.owner) is None:
        ui.clear_focus()
        return
    action: TextAction = handle_text_keys(ui.edit)
    if action is TextAction.editing:
        return
    if action is TextAction.cancel:
        ui.clear_focus()
        return
    commit_outliner_rename(scene, ui)


def sync_outliner_focus(scene: Scene, ui: UiState) -> None:
    """drop a rename edit when its group is gone"""
    if ui.focus is None or ui.focus.panel != outliner_panel:
        return
    if scene.nodes.get(ui.focus.owner) is None:
        ui.clear_focus()


def toggle_collapsed(ui: UiState, node_id: str) -> None:
    """expand or collapse a group in the tree"""
    if node_id in ui.outliner_collapsed:
        ui.outliner_collapsed.discard(node_id)
    else:
        ui.outliner_collapsed.add(node_id)


def click_row(scene: Scene, ui: UiState, node: Node) -> None:
    """select or rename a group, detecting a double-click on the same row"""
    now: float = get_time()
    is_double: bool = (
        ui.outliner_click_id == node.id
        and (now - ui.outliner_click_time) < ui_double_click_sec
    )
    ui.outliner_click_id = node.id
    ui.outliner_click_time = now
    if ui.focus is not None and ui.focus.panel == outliner_panel:
        commit_outliner_rename(scene, ui)
    if is_double:
        begin_outliner_rename(ui, node)
        return
    if ctrl_held():
        scene.toggle_selection(node.id)
    else:
        scene.set_selection([node.id])


def update_outliner(
    scene: Scene,
    ui: UiState,
    area: Rectangle,
    catalog: MeshCatalog,
    on_group: Callable[[], None],
    on_parent: Callable[[], None],
    on_unparent: Callable[[], None],
) -> tuple[list[OutlinerRow], list[Button]]:
    """lay out the header buttons and group rows; handle scroll, clicks, rename"""
    handle_outliner_typing(scene, ui)

    inner_w: float = area.width - ui_pad * 2.0
    title_h: float = float(ui_font_size + ui_pad)
    buttons_y: float = area.y + ui_pad + title_h
    btn_w: float = (inner_w - ui_button_gap * 2.0) / 3.0
    buttons: list[Button] = [
        Button(
            label="Group",
            on_click=on_group,
            rect=Rectangle(area.x + ui_pad, buttons_y, btn_w, float(ui_button_height)),
        ),
        Button(
            label="Parent",
            on_click=on_parent,
            rect=Rectangle(
                area.x + ui_pad + btn_w + ui_button_gap,
                buttons_y,
                btn_w,
                float(ui_button_height),
            ),
        ),
        Button(
            label="Unparent",
            on_click=on_unparent,
            rect=Rectangle(
                area.x + ui_pad + (btn_w + ui_button_gap) * 2.0,
                buttons_y,
                btn_w,
                float(ui_button_height),
            ),
        ),
    ]
    update_buttons(buttons, area)

    list_top: float = buttons_y + float(ui_button_height) + ui_button_gap
    list_h: float = max(0.0, area.y + area.height - ui_pad - list_top)
    list_rect: Rectangle = Rectangle(area.x, list_top, area.width, list_h)

    flat: list[tuple[Node, int, bool]] = flatten_groups(
        group_children_map(scene.nodes), ui.outliner_collapsed
    )
    content_h: float = len(flat) * outliner_row_height
    max_scroll: float = max(0.0, content_h - list_h)

    mouse: Vector2 = get_mouse_position()
    pointer_in_list: bool = is_point_in_rect(mouse.x, mouse.y, list_rect)
    if pointer_in_list:
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            ui.outliner_scroll -= wheel * outliner_row_height
    ui.outliner_scroll = max(0.0, min(max_scroll, ui.outliner_scroll))

    parent_target: str | None = scene.first_selected()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)

    rows: list[OutlinerRow] = []
    y: float = list_top - ui.outliner_scroll
    node: Node
    depth: int
    has_children: bool
    for node, depth, has_children in flat:
        rect: Rectangle = Rectangle(area.x, y, area.width, outliner_row_height)
        indent_x: float = area.x + ui_pad + depth * outliner_indent
        caret_rect: Rectangle = (
            Rectangle(indent_x, y, outliner_caret_width, outliner_row_height)
            if has_children
            else Rectangle(0.0, 0.0, 0.0, 0.0)
        )
        name_x: float = indent_x + outliner_caret_width
        name_rect: Rectangle = Rectangle(
            name_x,
            y,
            max(0.0, area.x + area.width - ui_pad - name_x),
            outliner_row_height,
        )
        row: OutlinerRow = OutlinerRow(
            node_id=node.id,
            depth=depth,
            name=group_display_name(node),
            summary=mesh_summary(scene.nodes, node.id, catalog),
            has_children=has_children,
            collapsed=node.id in ui.outliner_collapsed,
            rect=rect,
            caret_rect=caret_rect,
            name_rect=name_rect,
            is_selected=node.id in scene.selected_ids,
            is_parent_target=node.id == parent_target,
            is_hovered=is_point_in_rect(mouse.x, mouse.y, rect),
        )
        visible: bool = (
            rect.y + rect.height > list_rect.y
            and rect.y < list_rect.y + list_rect.height
        )
        if clicked and pointer_in_list and visible and row.is_hovered:
            if has_children and is_point_in_rect(mouse.x, mouse.y, caret_rect):
                toggle_collapsed(ui, node.id)
            else:
                click_row(scene, ui, node)
        rows.append(row)
        y += outliner_row_height

    return rows, buttons


def draw_outliner(
    font: Font,
    area: Rectangle,
    ui: UiState,
    rows: list[OutlinerRow],
    buttons: list[Button],
) -> None:
    """draw the outliner background, title, header buttons, and clipped rows"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x + area.width - 1, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        "Outliner",
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)

    if not rows or not buttons:
        return
    clip_top: float = buttons[0].rect.y + buttons[0].rect.height + ui_button_gap
    clip_h: float = max(0.0, area.y + area.height - ui_pad - clip_top)
    begin_scissor_mode(int(area.x), int(clip_top), int(area.width), int(clip_h))
    row: OutlinerRow
    for row in rows:
        draw_outliner_row(font, ui, row)
    end_scissor_mode()


def draw_outliner_row(font: Font, ui: UiState, row: OutlinerRow) -> None:
    """draw one tree row: caret, name (or rename field), and mesh summary"""
    if row.is_selected:
        draw_rectangle_rec(row.rect, ui_color_selection)
    elif row.is_hovered:
        draw_rectangle_rec(row.rect, ui_color_button_hover)
    if row.is_parent_target:
        draw_rectangle_lines_ex(row.rect, 2.0, ui_color_input_focus)

    size: float = float(ui_font_size)
    ty: float = row.rect.y + (row.rect.height - size) * 0.5
    if row.has_children:
        draw_text_ex(
            font,
            "v" if not row.collapsed else ">",
            Vector2(row.caret_rect.x + 2.0, ty),
            size,
            0,
            ui_color_text,
        )

    editing: bool = (
        ui.focus is not None
        and ui.edit is not None
        and ui.focus.panel == outliner_panel
        and ui.focus.owner == row.node_id
    )
    if editing and ui.edit is not None:
        draw_text_field(
            font,
            row.name_rect,
            ui.edit.text,
            focused=True,
            caret=ui.edit.caret,
            mark=ui.edit.mark,
        )
        return

    summary_size: float = float(ui_font_size - 4)
    summary_w: float = (
        measure_text_ex(font, row.summary, summary_size, 0).x if row.summary else 0.0
    )
    draw_text_ex(
        font,
        row.name,
        Vector2(row.name_rect.x, ty),
        size,
        0,
        ui_color_text,
    )
    if row.summary:
        draw_text_ex(
            font,
            row.summary,
            Vector2(
                row.rect.x + row.rect.width - ui_pad - summary_w,
                row.rect.y + (row.rect.height - summary_size) * 0.5,
            ),
            summary_size,
            0,
            ui_color_text,
        )
