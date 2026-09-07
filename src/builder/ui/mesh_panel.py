"""mesh catalog panel: list assets and set the active mesh"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    is_mouse_button_released,
    measure_text_ex,
)

from builder.meshes.mesh_catalog import MeshAsset, MeshAssetKind, MeshCatalog
from builder.scene.scene_types import MeshId
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
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_buttons


@dataclass
class MeshRow:
    mesh_id: MeshId
    label: str
    detail: str
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False
    is_selected: bool = False


def mesh_asset_detail(asset: MeshAsset) -> str:
    """short secondary line for a catalog asset"""
    match asset.kind:
        case MeshAssetKind.builtin:
            return MeshAssetKind.builtin.name
        case MeshAssetKind.fbx:
            if asset.source_path is None:
                return MeshAssetKind.fbx.name
            return Path(asset.source_path).name


def update_mesh_panel(
    catalog: MeshCatalog,
    active_mesh_id: MeshId,
    ui: UiState,
    area: Rectangle,
    on_select: Callable[[MeshId], None],
    on_import: Callable[[], None],
) -> tuple[list[MeshRow], list[Button]]:
    """layout rows and import button; handle scroll/clicks"""
    assets: list[MeshAsset] = list(catalog.entries.values())
    title_h: float = float(ui_font_size + ui_pad)
    import_rect: Rectangle = Rectangle(
        area.x + ui_pad,
        area.y + ui_pad + title_h,
        area.width - ui_pad * 2.0,
        float(ui_button_height),
    )
    list_top: float = import_rect.y + import_rect.height + ui_button_gap
    list_h: float = max(0.0, area.y + area.height - ui_pad - list_top)
    list_rect: Rectangle = Rectangle(area.x, list_top, area.width, list_h)

    row_h: float = float(ui_button_height + 10)
    content_h: float = max(0.0, len(assets) * (row_h + ui_button_gap) - ui_button_gap)
    max_scroll: float = max(0.0, content_h - list_h)

    mouse: Vector2 = get_mouse_position()
    if is_point_in_rect(mouse.x, mouse.y, list_rect):
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            ui.meshes_panel_scroll -= wheel * row_h
    ui.meshes_panel_scroll = max(0.0, min(max_scroll, ui.meshes_panel_scroll))

    rows: list[MeshRow] = []
    y: float = list_top - ui.meshes_panel_scroll
    bw: float = area.width - ui_pad * 2.0
    left_down: bool = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    pointer_in_list: bool = is_point_in_rect(mouse.x, mouse.y, list_rect)

    asset: MeshAsset
    for asset in assets:
        row: MeshRow = MeshRow(
            mesh_id=asset.mesh_id,
            label=asset.label,
            detail=mesh_asset_detail(asset),
            rect=Rectangle(area.x + ui_pad, y, bw, row_h),
            is_selected=asset.mesh_id == active_mesh_id,
        )
        row.is_hovered = is_point_in_rect(mouse.x, mouse.y, row.rect)
        row.is_pressed = row.is_hovered and left_down
        visible: bool = (
            row.rect.y + row.rect.height > list_rect.y
            and row.rect.y < list_rect.y + list_rect.height
        )
        if pointer_in_list and visible and row.is_hovered and left_released:
            on_select(asset.mesh_id)
        rows.append(row)
        y += row_h + ui_button_gap

    buttons: list[Button] = [
        Button(label="Import mesh", on_click=on_import, rect=import_rect)
    ]
    update_buttons(buttons, area)
    return rows, buttons


def draw_mesh_row(font: Font, row: MeshRow) -> None:
    """draw one mesh catalog row"""
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
    label: str = marker + row.label
    label_size: float = float(ui_font_size)
    detail_size: float = float(ui_font_size - 4)
    tx: float = row.rect.x + 8.0
    ty: float = row.rect.y + 4.0
    draw_text_ex(font, label, Vector2(tx, ty), label_size, 0, ui_color_text)
    detail_w: float = measure_text_ex(font, row.detail, detail_size, 0).x
    max_detail_w: float = max(0.0, row.rect.width - 16.0)
    detail: str = row.detail
    if detail_w > max_detail_w and len(detail) > 3:
        # trim with ellipsis so long paths fit the row
        while (
            len(detail) > 3
            and measure_text_ex(font, detail + "...", detail_size, 0).x > max_detail_w
        ):
            detail = detail[:-1]
        detail = detail + "..."
    draw_text_ex(
        font,
        detail,
        Vector2(tx + 14.0, ty + label_size),
        detail_size,
        0,
        ui_color_text,
    )


def draw_mesh_panel(
    font: Font,
    area: Rectangle,
    rows: list[MeshRow],
    buttons: list[Button],
) -> None:
    """draw the meshes panel background, title, import button, and clipped rows"""
    draw_rectangle_rec(area, ui_color_panel)
    draw_rectangle_lines_ex(
        Rectangle(area.x, area.y, 1, area.height),
        1,
        ui_color_border,
    )
    draw_text_ex(
        font,
        "Meshes",
        Vector2(area.x + ui_pad, area.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    button: Button
    for button in buttons:
        draw_button(button, font)
    title_h: float = float(ui_font_size + ui_pad)
    list_top: float = (
        area.y + ui_pad + title_h + float(ui_button_height) + ui_button_gap
    )
    list_h: float = max(0.0, area.y + area.height - ui_pad - list_top)
    begin_scissor_mode(int(area.x), int(list_top), int(area.width), int(list_h))
    row: MeshRow
    for row in rows:
        draw_mesh_row(font, row)
    end_scissor_mode()
