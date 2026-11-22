"""inspector for a selected vector layer: add shapes by kind"""

from __future__ import annotations

from pyray import (
    Font,
    Rectangle,
    Vector2,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
)

from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.vector.coords import vector_layer_is_road
from builder.heightfield.vector.edit import add_vector_shape
from builder.heightfield.vector.rasterize import (
    find_heightfield_group_for_descendant,
    regenerate_vector_layer_for_node,
)
from builder.heightfield.vector.registry import vector_shape_specs
from builder.heightfield.vector.vector_types import VectorShapeSpec
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_panel,
    ui_color_text,
    ui_font_size,
    ui_pad,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import Button, draw_button, update_buttons


def selected_vector_layer_group(scene: Scene) -> Node | None:
    """return the sole selected node if it owns a vector layer recipe"""
    if len(scene.selected_ids) != 1:
        return None
    node: Node | None = scene.nodes.get(scene.selected_ids[0])
    if node is None or node.vector_layer is None:
        return None
    return node


def update_vector_layer_inspector(
    scene: Scene,
    catalog: HeightmapCatalog,
    ui: UiState,
    area: Rectangle,
    layer_node: Node,
) -> list[Button]:
    """layout/input: one add button per registered shape kind"""
    if layer_node.vector_layer is None:
        return []
    buttons: list[Button] = []
    x: float = area.x + ui_pad
    y: float = area.y + ui_pad + float(ui_font_size) + ui_button_gap
    inner_w: float = area.width - 2.0 * ui_pad
    row_h: float = float(ui_button_height)
    layer_id: str = layer_node.id
    road_layer: bool = False
    hf_group_id: str | None = find_heightfield_group_for_descendant(
        scene.nodes, layer_id
    )
    if hf_group_id is not None:
        hf_node: Node | None = scene.nodes.get(hf_group_id)
        if hf_node is not None and hf_node.heightfield is not None:
            road_layer = vector_layer_is_road(hf_node.heightfield, layer_id)

    spec: VectorShapeSpec
    for spec in vector_shape_specs:

        def add_shape(
            kind: str = spec.kind,
            label: str = spec.label,
            as_road: bool = road_layer,
        ) -> None:
            ui.clear_focus()
            new_nodes: list[Node] = add_vector_shape(
                layer_id, kind, road=as_road
            )
            scene.add_nodes(new_nodes)
            hf_id: str | None = find_heightfield_group_for_descendant(
                scene.nodes, layer_id
            )
            if hf_id is not None:
                regenerate_vector_layer_for_node(
                    catalog, scene.nodes, hf_id, layer_id
                )
            shape_id: str = new_nodes[0].id
            scene.set_selection([shape_id])
            ui.status = f"Added {label}"

        buttons.append(
            Button(
                label=f"+ {spec.label}",
                on_click=add_shape,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    update_buttons(buttons, area)
    return buttons


def draw_vector_layer_inspector(
    font: Font,
    area: Rectangle,
    layer_node: Node,
    buttons: list[Button],
) -> None:
    """draw vector layer inspector chrome and add-shape buttons"""
    title: str = layer_node.name if layer_node.name != "" else "Vector"
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
    button: Button
    for button in buttons:
        draw_button(button, font)
