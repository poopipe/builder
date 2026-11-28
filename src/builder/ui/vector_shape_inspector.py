"""inspector controls for a selected vector shape node"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Transform,
    Vector2,
    Vector3,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_pressed,
)

from builder.heightfield.brushes.brush_types import BrushSpec
from builder.heightfield.brushes.registry import vector_brush_specs
from builder.heightfield.gpu.heightfield_gpu import HeightfieldGpuState
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.vector.circle import CircleShapeParams
from builder.heightfield.vector.coords import (
    vector_layer_is_road,
    vector_point_belongs_to_road,
)
from builder.heightfield.vector.edit import append_polyline_point
from builder.heightfield.vector.polyline import PolylineShapeParams
from builder.heightfield.vector.rasterize import (
    dirty_aabb_for_vector_nodes,
    find_heightfield_group_for_descendant,
    find_vector_layer_ancestor,
    find_vector_shape_ancestor,
    regenerate_vectors_touching,
)
from builder.heightfield.vector.registry import get_vector_shape_spec
from builder.heightfield.vector.stamp import (
    is_absolute_flatten_brush,
    is_flatten_brush,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorShapeRecipe,
)
from builder.path.path_types import PathPointRecipe
from builder.path.points import (
    default_path_point_recipe,
    ordered_path_points,
    path_point_width_scale,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node, NodeRole
from builder.ui.list_picker import ListPickerPurpose, make_list_picker
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
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_buttons

vector_shape_panel_id: str = "vector_shape"


@dataclass(frozen=True)
class ShapeFloatField:
    """one float exposed in the shape inspector"""

    key: str
    title: str
    step: float


@dataclass
class VectorShapeFloatRow:
    """± / type-in for a shape float"""

    key: str
    title: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle
    # when set, type-in focus/draw use this owner and value (road point height)
    owner: str | None = None
    display_value: float | None = None


def selected_vector_shape_group(scene: Scene) -> Node | None:
    """return selected vector shape, or the shape owning a selected vector point"""
    if len(scene.selected_ids) != 1:
        return None
    node: Node | None = scene.nodes.get(scene.selected_ids[0])
    if node is None:
        return None
    if node.vector_shape is not None:
        return node
    if node.role is not NodeRole.path_point:
        return None
    shape_id: str | None = find_vector_shape_ancestor(scene.nodes, node.id)
    if shape_id is None:
        return None
    shape: Node | None = scene.nodes.get(shape_id)
    if shape is None or shape.vector_shape is None:
        return None
    return shape


def format_shape_float(value: float) -> str:
    """format a shape float for the type-in field"""
    return f"{float(value):.4g}"


def parse_shape_float(text: str) -> float | None:
    """parse typed shape float text"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def clamp_shape_float(key: str, value: float) -> float:
    """clamp shape float keys to valid ranges"""
    if key == "grade":
        return max(0.0, min(1.0, value))
    if key in ("offset", "intensity"):
        return value
    return max(0.1, value)


def shape_belongs_to_road(
    scene: Scene, shape_node: Node
) -> bool:
    """true when the shape's vector layer is a road height source"""
    layer_id: str | None = find_vector_layer_ancestor(
        scene.nodes, shape_node.id
    )
    hf_id: str | None = find_heightfield_group_for_descendant(
        scene.nodes, shape_node.id
    )
    if layer_id is None or hf_id is None:
        return False
    hf_node: Node | None = scene.nodes.get(hf_id)
    if hf_node is None or hf_node.heightfield is None:
        return False
    return vector_layer_is_road(hf_node.heightfield, layer_id)


def shape_float_fields(
    recipe: VectorShapeRecipe, *, road: bool = False
) -> list[ShapeFloatField]:
    """ordered float fields for this recipe (tab cycle)"""
    fields: list[ShapeFloatField] = []
    if road:
        fields.append(ShapeFloatField("intensity", "Intensity", 0.05))
        fields.append(ShapeFloatField("grade", "Grade", 0.05))
        fields.append(ShapeFloatField("radius", "Width", 0.5))
    else:
        offset_label: str = (
            "Y" if is_absolute_flatten_brush(recipe.brush_kind) else "Offset"
        )
        fields.append(ShapeFloatField("offset", offset_label, 0.25))
        if is_flatten_brush(recipe.brush_kind):
            fields.append(ShapeFloatField("intensity", "Intensity", 0.05))
        fields.append(ShapeFloatField("radius", "Brush radius", 0.5))
    match recipe.params:
        case CircleShapeParams():
            fields.append(ShapeFloatField("shape_radius", "Shape radius", 0.5))
        case _:
            pass
    return fields


def read_shape_float(recipe: VectorShapeRecipe, key: str) -> float | None:
    """read one inspector float from the recipe"""
    if key == "offset":
        return float(recipe.offset.value)
    if key == "intensity":
        return float(recipe.intensity.value)
    if key == "radius":
        return float(recipe.radius.value)
    if key == "grade":
        return float(recipe.grade.value)
    if key == "shape_radius":
        match recipe.params:
            case CircleShapeParams() as circle:
                return float(circle.radius.value)
            case _:
                return None
    return None


def apply_shape_float(
    recipe: VectorShapeRecipe, key: str, value: float
) -> VectorShapeRecipe | None:
    """return recipe with one float replaced, or none if key does not apply"""
    clamped: float = clamp_shape_float(key, value)
    if key == "offset":
        return replace(recipe, offset=replace(recipe.offset, value=clamped))
    if key == "intensity":
        return replace(
            recipe, intensity=replace(recipe.intensity, value=clamped)
        )
    if key == "radius":
        return replace(recipe, radius=replace(recipe.radius, value=clamped))
    if key == "grade":
        return replace(recipe, grade=replace(recipe.grade, value=clamped))
    if key == "shape_radius":
        match recipe.params:
            case CircleShapeParams() as circle:
                return replace(
                    recipe,
                    params=replace(
                        circle, radius=replace(circle.radius, value=clamped)
                    ),
                )
            case _:
                return None
    return None

def commit_shape_recipe(
    scene: Scene,
    catalog: HeightmapCatalog,
    shape_node: Node,
    recipe: VectorShapeRecipe,
    ui: UiState,
    *,
    heightfield_gpu: HeightfieldGpuState | None = None,
) -> None:
    """replace shape recipe and re-rasterize its vector layer"""
    prior = dirty_aabb_for_vector_nodes(scene.nodes, [shape_node.id])
    updated: Node = replace(shape_node, vector_shape=recipe)
    scene.add_nodes([updated])
    regenerate_vectors_touching(
        catalog,
        scene.nodes,
        [updated.id],
        prior_dirty_aabb=prior,
        defer_gpu=False,
        heightfield_gpu=heightfield_gpu,
    )
    ui.status = f"Updated {recipe.kind}"


def next_apply_mode(current: VectorApplyMode) -> VectorApplyMode:
    """cycle stroke/fill"""
    modes: list[VectorApplyMode] = list(VectorApplyMode)
    return modes[(modes.index(current) + 1) % len(modes)]


def vector_brush_label(kind: str) -> str:
    """display label for a vector sculpt brush kind"""
    spec: BrushSpec
    for spec in vector_brush_specs:
        if spec.kind == kind:
            return spec.label
    return kind


def vector_brush_kind_for_label(label: str) -> str | None:
    """map a picker label back to brush kind"""
    spec: BrushSpec
    for spec in vector_brush_specs:
        if spec.label == label:
            return spec.kind
    return None


def apply_vector_brush_choice(
    scene: Scene,
    catalog: HeightmapCatalog,
    ui: UiState,
    shape_id: str,
    chosen_label: str,
    *,
    heightfield_gpu: HeightfieldGpuState | None = None,
) -> None:
    """commit a list-picker brush choice onto a vector shape"""
    kind: str | None = vector_brush_kind_for_label(chosen_label)
    if kind is None:
        ui.status = f"Unknown brush: {chosen_label}"
        return
    shape_node: Node | None = scene.nodes.get(shape_id)
    if shape_node is None or shape_node.vector_shape is None:
        ui.status = "Brush target missing"
        return
    recipe: VectorShapeRecipe = shape_node.vector_shape
    if recipe.brush_kind == kind:
        ui.status = f"Brush: {chosen_label}"
        return
    commit_shape_recipe(
        scene,
        catalog,
        shape_node,
        replace(recipe, brush_kind=kind),
        ui,
        heightfield_gpu=heightfield_gpu,
    )
    ui.status = f"Brush: {chosen_label}"


def neighbor_float_key(
    fields: list[ShapeFloatField], key: str, delta: int
) -> str:
    """wrap to the next/prev float field key"""
    keys: list[str] = [field.key for field in fields]
    if key not in keys:
        return keys[0]
    index: int = keys.index(key)
    return keys[(index + delta) % len(keys)]


def update_vector_shape_inspector(
    scene: Scene,
    catalog: HeightmapCatalog,
    ui: UiState,
    area: Rectangle,
    shape_node: Node,
    *,
    heightfield_gpu: HeightfieldGpuState | None = None,
) -> tuple[list[Button], list[VectorShapeFloatRow]]:
    """layout/input for vector shape inspector"""
    recipe: VectorShapeRecipe | None = shape_node.vector_shape
    if recipe is None:
        return ([], [])
    buttons: list[Button] = []
    float_rows: list[VectorShapeFloatRow] = []
    x: float = area.x + ui_pad
    y: float = area.y + ui_pad + float(ui_font_size) + ui_button_gap
    inner_w: float = area.width - 2.0 * ui_pad
    row_h: float = float(ui_button_height)
    step_w: float = float(ui_stepper_width)
    mouse: Vector2 = get_mouse_position()
    clicked: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    owner: str = shape_node.id
    road: bool = shape_belongs_to_road(scene, shape_node)
    fields: list[ShapeFloatField] = shape_float_fields(recipe, road=road)

    if ui.focus is not None and ui.focus.panel == vector_shape_panel_id:
        if ui.edit is not None and ui.focus.owner == owner:
            action: TextAction = handle_text_keys(ui.edit)
            match action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    parsed: float | None = parse_shape_float(ui.edit.text)
                    if parsed is not None:
                        updated_recipe: VectorShapeRecipe | None = (
                            apply_shape_float(recipe, ui.focus.key, parsed)
                        )
                        if updated_recipe is not None:
                            commit_shape_recipe(
                                scene,
                                catalog,
                                shape_node,
                                updated_recipe,
                                ui,
                                heightfield_gpu=heightfield_gpu,
                            )
                        live: Node | None = scene.nodes.get(shape_node.id)
                        if live is not None and live.vector_shape is not None:
                            shape_node = live
                            recipe = live.vector_shape
                            road = shape_belongs_to_road(scene, shape_node)
                            fields = shape_float_fields(recipe, road=road)
                    if recipe is None:
                        return ([], [])
                    if action is TextAction.commit:
                        ui.clear_focus()
                    elif action is TextAction.focus_next:
                        next_key: str = neighbor_float_key(
                            fields, ui.focus.key, 1
                        )
                        next_value: float | None = read_shape_float(
                            recipe, next_key
                        )
                        if next_value is not None:
                            ui.focus = FieldId(
                                panel=vector_shape_panel_id,
                                key=next_key,
                                owner=owner,
                            )
                            ui.edit = begin_edit(
                                format_shape_float(next_value)
                            )
                    elif action is TextAction.focus_prev:
                        prev_key: str = neighbor_float_key(
                            fields, ui.focus.key, -1
                        )
                        prev_value: float | None = read_shape_float(
                            recipe, prev_key
                        )
                        if prev_value is not None:
                            ui.focus = FieldId(
                                panel=vector_shape_panel_id,
                                key=prev_key,
                                owner=owner,
                            )
                            ui.edit = begin_edit(
                                format_shape_float(prev_value)
                            )

    live_node: Node | None = scene.nodes.get(shape_node.id)
    if live_node is not None and live_node.vector_shape is not None:
        shape_node = live_node
        recipe = live_node.vector_shape
        road = shape_belongs_to_road(scene, shape_node)
        fields = shape_float_fields(recipe, road=road)
    if recipe is None:
        return ([], [])

    def set_apply() -> None:
        ui.clear_focus()
        commit_shape_recipe(
            scene,
            catalog,
            shape_node,
            replace(recipe, apply_mode=next_apply_mode(recipe.apply_mode)),
            ui,
            heightfield_gpu=heightfield_gpu,
        )

    def open_brush_picker() -> None:
        ui.clear_focus()
        if ui.file_browser is not None:
            return
        labels: list[str] = [spec.label for spec in vector_brush_specs]
        if not labels:
            return
        current_label: str = vector_brush_label(recipe.brush_kind)
        index: int
        try:
            index = labels.index(current_label)
        except ValueError:
            index = 0
        ui.list_picker = make_list_picker(
            purpose=ListPickerPurpose.vector_brush,
            title="Select brush",
            items=labels,
            selected=index,
            target_id=shape_node.id,
        )
        ui.status = "Select brush…"

    def toggle_enabled() -> None:
        ui.clear_focus()
        commit_shape_recipe(
            scene,
            catalog,
            shape_node,
            replace(recipe, enabled=not recipe.enabled),
            ui,
            heightfield_gpu=heightfield_gpu,
        )

    buttons.append(
        Button(
            label=f"Apply: {recipe.apply_mode.name}",
            on_click=set_apply,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap
    if not road:
        buttons.append(
            Button(
                label=f"Brush: {vector_brush_label(recipe.brush_kind)}",
                on_click=open_brush_picker,
                rect=Rectangle(x, y, inner_w, row_h),
            )
        )
        y += row_h + ui_button_gap

    road_point: Node | None = None
    if len(scene.selected_ids) == 1:
        point_cand: Node | None = scene.nodes.get(scene.selected_ids[0])
        if (
            point_cand is not None
            and point_cand.role is NodeRole.path_point
            and vector_point_belongs_to_road(scene.nodes, point_cand.id)
        ):
            road_point = point_cand

    if road_point is not None:
        height_key: str = "point_height"
        height_label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        height_controls_y: float = y + float(ui_font_size) + 4.0
        height_minus: Rectangle = Rectangle(x, height_controls_y, step_w, row_h)
        height_plus: Rectangle = Rectangle(
            x + inner_w - step_w, height_controls_y, step_w, row_h
        )
        height_value: Rectangle = Rectangle(
            height_minus.x + height_minus.width + ui_button_gap,
            height_controls_y,
            height_plus.x
            - (height_minus.x + height_minus.width + ui_button_gap * 2),
            row_h,
        )
        float_rows.append(
            VectorShapeFloatRow(
                key=height_key,
                title="Height offset",
                label=height_label,
                minus=height_minus,
                value=height_value,
                plus=height_plus,
                owner=road_point.id,
                display_value=float(road_point.transform.translation.y),
            )
        )

        def commit_point_height(new_y: float, point_id: str = road_point.id) -> None:
            live_point: Node | None = scene.nodes.get(point_id)
            if live_point is None:
                return
            prior = dirty_aabb_for_vector_nodes(scene.nodes, [point_id])
            translation: Vector3 = live_point.transform.translation
            scene.set_node(
                replace(
                    live_point,
                    transform=Transform(
                        Vector3(
                            float(translation.x),
                            float(new_y),
                            float(translation.z),
                        ),
                        live_point.transform.rotation,
                        live_point.transform.scale,
                    ),
                )
            )
            regenerate_vectors_touching(
                catalog,
                scene.nodes,
                [point_id],
                prior_dirty_aabb=prior,
                defer_gpu=False,
                heightfield_gpu=heightfield_gpu,
            )
            ui.status = "Updated height offset"

        def nudge_height_minus() -> None:
            ui.clear_focus()
            live_point = scene.nodes.get(road_point.id)
            if live_point is None:
                return
            commit_point_height(
                float(live_point.transform.translation.y) - 0.5
            )

        def nudge_height_plus() -> None:
            ui.clear_focus()
            live_point = scene.nodes.get(road_point.id)
            if live_point is None:
                return
            commit_point_height(
                float(live_point.transform.translation.y) + 0.5
            )

        buttons.append(
            Button(label="-", on_click=nudge_height_minus, rect=height_minus)
        )
        buttons.append(
            Button(label="+", on_click=nudge_height_plus, rect=height_plus)
        )
        if clicked and is_point_in_rect(mouse.x, mouse.y, height_value):
            ui.focus = FieldId(
                panel=vector_shape_panel_id,
                key=height_key,
                owner=road_point.id,
            )
            ui.edit = begin_edit(
                format_shape_float(
                    float(road_point.transform.translation.y)
                )
            )
        y = height_controls_y + row_h + ui_button_gap

        if (
            ui.focus is not None
            and ui.focus.panel == vector_shape_panel_id
            and ui.focus.key == height_key
            and ui.focus.owner == road_point.id
            and ui.edit is not None
        ):
            height_action: TextAction = handle_text_keys(ui.edit)
            match height_action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    height_parsed: float | None = parse_shape_float(
                        ui.edit.text
                    )
                    if height_parsed is not None:
                        commit_point_height(height_parsed)
                    ui.clear_focus()

        scale_key: str = "point_width_scale"
        scale_value: float = path_point_width_scale(road_point)
        scale_label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
        scale_controls_y: float = y + float(ui_font_size) + 4.0
        scale_minus: Rectangle = Rectangle(x, scale_controls_y, step_w, row_h)
        scale_plus: Rectangle = Rectangle(
            x + inner_w - step_w, scale_controls_y, step_w, row_h
        )
        scale_value_rect: Rectangle = Rectangle(
            scale_minus.x + scale_minus.width + ui_button_gap,
            scale_controls_y,
            scale_plus.x
            - (scale_minus.x + scale_minus.width + ui_button_gap * 2),
            row_h,
        )
        float_rows.append(
            VectorShapeFloatRow(
                key=scale_key,
                title="Width scale",
                label=scale_label,
                minus=scale_minus,
                value=scale_value_rect,
                plus=scale_plus,
                owner=road_point.id,
                display_value=scale_value,
            )
        )

        def commit_point_width_scale(
            new_scale: float, point_id: str = road_point.id
        ) -> None:
            live_point: Node | None = scene.nodes.get(point_id)
            if live_point is None:
                return
            prior = dirty_aabb_for_vector_nodes(scene.nodes, [point_id])
            recipe: PathPointRecipe = (
                live_point.path_point
                if live_point.path_point is not None
                else default_path_point_recipe()
            )
            clamped: float = max(0.05, float(new_scale))
            scene.set_node(
                replace(
                    live_point,
                    path_point=replace(
                        recipe,
                        width_scale=replace(
                            recipe.width_scale, value=clamped
                        ),
                    ),
                )
            )
            regenerate_vectors_touching(
                catalog,
                scene.nodes,
                [point_id],
                prior_dirty_aabb=prior,
                defer_gpu=False,
                heightfield_gpu=heightfield_gpu,
            )
            ui.status = "Updated width scale"

        def nudge_scale_minus() -> None:
            ui.clear_focus()
            live_point = scene.nodes.get(road_point.id)
            if live_point is None:
                return
            commit_point_width_scale(
                path_point_width_scale(live_point) - 0.05
            )

        def nudge_scale_plus() -> None:
            ui.clear_focus()
            live_point = scene.nodes.get(road_point.id)
            if live_point is None:
                return
            commit_point_width_scale(
                path_point_width_scale(live_point) + 0.05
            )

        buttons.append(
            Button(label="-", on_click=nudge_scale_minus, rect=scale_minus)
        )
        buttons.append(
            Button(label="+", on_click=nudge_scale_plus, rect=scale_plus)
        )
        if clicked and is_point_in_rect(mouse.x, mouse.y, scale_value_rect):
            ui.focus = FieldId(
                panel=vector_shape_panel_id,
                key=scale_key,
                owner=road_point.id,
            )
            ui.edit = begin_edit(format_shape_float(scale_value))
        y = scale_controls_y + row_h + ui_button_gap

        if (
            ui.focus is not None
            and ui.focus.panel == vector_shape_panel_id
            and ui.focus.key == scale_key
            and ui.focus.owner == road_point.id
            and ui.edit is not None
        ):
            scale_action: TextAction = handle_text_keys(ui.edit)
            match scale_action:
                case TextAction.editing:
                    pass
                case TextAction.cancel:
                    ui.clear_focus()
                case TextAction.commit | TextAction.focus_next | TextAction.focus_prev:
                    scale_parsed: float | None = parse_shape_float(
                        ui.edit.text
                    )
                    if scale_parsed is not None:
                        commit_point_width_scale(scale_parsed)
                    ui.clear_focus()

    field: ShapeFloatField
    for field in fields:
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
            VectorShapeFloatRow(
                key=field.key,
                title=field.title,
                label=label_rect,
                minus=minus_rect,
                value=value_rect,
                plus=plus_rect,
            )
        )

        def nudge_minus(
            k: str = field.key, s: float = field.step
        ) -> None:
            ui.clear_focus()
            current: Node | None = scene.nodes.get(shape_node.id)
            if current is None or current.vector_shape is None:
                return
            current_value: float | None = read_shape_float(
                current.vector_shape, k
            )
            if current_value is None:
                return
            updated: VectorShapeRecipe | None = apply_shape_float(
                current.vector_shape, k, current_value - s
            )
            if updated is None:
                return
            commit_shape_recipe(
                scene,
                catalog,
                current,
                updated,
                ui,
                heightfield_gpu=heightfield_gpu,
            )

        def nudge_plus(
            k: str = field.key, s: float = field.step
        ) -> None:
            ui.clear_focus()
            current = scene.nodes.get(shape_node.id)
            if current is None or current.vector_shape is None:
                return
            current_value = read_shape_float(current.vector_shape, k)
            if current_value is None:
                return
            updated = apply_shape_float(
                current.vector_shape, k, current_value + s
            )
            if updated is None:
                return
            commit_shape_recipe(
                scene,
                catalog,
                current,
                updated,
                ui,
                heightfield_gpu=heightfield_gpu,
            )

        buttons.append(Button(label="-", on_click=nudge_minus, rect=minus_rect))
        buttons.append(Button(label="+", on_click=nudge_plus, rect=plus_rect))
        if clicked and is_point_in_rect(mouse.x, mouse.y, value_rect):
            current_value = read_shape_float(recipe, field.key)
            if current_value is not None:
                ui.focus = FieldId(
                    panel=vector_shape_panel_id, key=field.key, owner=owner
                )
                ui.edit = begin_edit(format_shape_float(current_value))
        y = controls_y + row_h + ui_button_gap

    buttons.append(
        Button(
            label=("Enabled" if recipe.enabled else "Disabled"),
            on_click=toggle_enabled,
            rect=Rectangle(x, y, inner_w, row_h),
        )
    )
    y += row_h + ui_button_gap

    match recipe.params:
        case PolylineShapeParams() as poly:
            def toggle_closed() -> None:
                ui.clear_focus()
                current = scene.nodes.get(shape_node.id)
                if current is None or current.vector_shape is None:
                    return
                r: VectorShapeRecipe = current.vector_shape
                match r.params:
                    case PolylineShapeParams() as p:
                        new_params = replace(
                            p, closed=replace(p.closed, value=not p.closed.value)
                        )
                        commit_shape_recipe(
                            scene,
                            catalog,
                            current,
                            replace(r, params=new_params),
                            ui,
                            heightfield_gpu=heightfield_gpu,
                        )
                    case _:
                        pass

            buttons.append(
                Button(
                    label=("Closed" if poly.closed.value else "Open"),
                    on_click=toggle_closed,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
            y += row_h + ui_button_gap

            def add_point() -> None:
                ui.clear_focus()
                existing_points = ordered_path_points(
                    scene.nodes, shape_node.id
                )
                prior_ids: list[str] = (
                    [existing_points[-1].id]
                    if existing_points
                    else [shape_node.id]
                )
                prior = dirty_aabb_for_vector_nodes(scene.nodes, prior_ids)
                created: list[Node] = append_polyline_point(
                    scene.nodes, shape_node.id
                )
                scene.add_nodes(created)
                point_id: str = created[0].id
                created_node: Node
                for created_node in created:
                    if created_node.role is NodeRole.path_point:
                        point_id = created_node.id
                        break
                regenerate_vectors_touching(
                    catalog,
                    scene.nodes,
                    [point_id],
                    prior_dirty_aabb=prior,
                    defer_gpu=False,
                    heightfield_gpu=heightfield_gpu,
                )
                scene.set_selection([point_id])
                ui.status = "Added point"

            buttons.append(
                Button(
                    label="+ Point",
                    on_click=add_point,
                    rect=Rectangle(x, y, inner_w, row_h),
                )
            )
        case _:
            pass

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
    return buttons, float_rows


def draw_vector_shape_inspector(
    font: Font,
    area: Rectangle,
    ui: UiState,
    shape_node: Node,
    buttons: list[Button],
    float_rows: list[VectorShapeFloatRow],
) -> None:
    """draw vector shape inspector chrome, floats, and buttons"""
    recipe: VectorShapeRecipe | None = shape_node.vector_shape
    title: str = "Vector Shape"
    if recipe is not None:
        try:
            title = get_vector_shape_spec(recipe.kind).label
        except KeyError:
            title = recipe.kind
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
    if recipe is None:
        return
    float_row: VectorShapeFloatRow
    for float_row in float_rows:
        draw_text_ex(
            font,
            float_row.title,
            Vector2(float_row.label.x, float_row.label.y),
            float(ui_font_size),
            0,
            ui_color_text,
        )
        row_owner: str = (
            float_row.owner if float_row.owner is not None else shape_node.id
        )
        edit: TextEdit | None = ui.focused_edit(
            vector_shape_panel_id, float_row.key, owner=row_owner
        )
        value: float | None = float_row.display_value
        if value is None:
            value = read_shape_float(recipe, float_row.key)
        if value is None:
            continue
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
                format_shape_float(value),
                focused=False,
                center_unfocused=True,
            )
