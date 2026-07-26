"""menu bar type-in transform for the selected groups"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pyray import (
    Font,
    MouseButton,
    Rectangle,
    Vector2,
    draw_text_ex,
    get_mouse_position,
    is_mouse_button_pressed,
    measure_text_ex,
)

from builder.scene.scene import Scene
from builder.scene.scene_types import Node
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
    ui_color_text,
    ui_font_size,
    ui_pad,
)
from builder.ui.transform_fields import (
    apply_transform_field,
    field_index,
    format_transform_value,
    parse_transform_value,
    selected_nodes,
    transform_field_keys,
    transform_field_labels,
    transform_field_values,
)
from builder.ui.ui_state import UiState
from builder.ui.widgets import is_point_in_rect
from builder.view.gizmo_types import GizmoMode, GizmoSpace

transform_panel: str = "transform"
transform_field_width: float = 58.0
transform_label_gap: float = 4.0


@dataclass(frozen=True)
class TransformField:
    """one laid-out type-in field"""

    key: str
    label: str
    label_rect: Rectangle
    rect: Rectangle


@dataclass(frozen=True)
class TransformBarRects:
    """laid-out type-in transform for one frame"""

    area: Rectangle
    owner: str
    fields: tuple[TransformField, ...]


def layout_transform_bar(
    font: Font,
    area: Rectangle,
    scene: Scene,
    mode: GizmoMode,
    min_x: float,
) -> TransformBarRects | None:
    """place fields at the right of the menu bar; None if they will not fit"""
    nodes: list[Node] = selected_nodes(scene)
    if not nodes:
        return None

    size: float = float(ui_font_size)
    labels: tuple[str, str, str] = transform_field_labels(mode)
    keys: tuple[str, str, str] = transform_field_keys(mode)
    label_w: float = max(
        measure_text_ex(font, label, size, 0).x for label in labels
    )
    group_w: float = (
        (label_w + transform_label_gap + transform_field_width) * 3.0
        + ui_button_gap * 2.0
    )
    height: float = float(ui_button_height)
    x: float = area.x + area.width - ui_pad - group_w
    if x < min_x:
        return None
    y: float = area.y + (area.height - height) * 0.5

    fields: list[TransformField] = []
    index: int
    for index in range(3):
        label_rect: Rectangle = Rectangle(x, y, label_w, height)
        field_rect: Rectangle = Rectangle(
            x + label_w + transform_label_gap,
            y,
            transform_field_width,
            height,
        )
        fields.append(
            TransformField(
                key=keys[index],
                label=labels[index],
                label_rect=label_rect,
                rect=field_rect,
            )
        )
        x = field_rect.x + field_rect.width + ui_button_gap
    return TransformBarRects(
        area=area,
        owner=nodes[0].id,
        fields=tuple(fields),
    )


def clear_transform_focus(ui: UiState) -> None:
    """drop editing when the fields are off screen, so Escape works again"""
    if ui.focus is not None and ui.focus.panel == transform_panel:
        ui.clear_focus()


def sync_transform_focus(ui: UiState, owner: str, keys: Sequence[str]) -> None:
    """drop editing when the selection or the tool changes under it"""
    if ui.focus is None or ui.focus.panel != transform_panel:
        return
    if ui.focus.owner != owner or ui.focus.key not in keys:
        ui.clear_focus()


def focus_transform_field(
    scene: Scene,
    ui: UiState,
    mode: GizmoMode,
    space: GizmoSpace,
    owner: str,
    key: str,
) -> None:
    """begin editing one field, seeded with its current value"""
    index: int | None = field_index(transform_field_keys(mode), key)
    if index is None:
        return
    values: tuple[float, float, float] = transform_field_values(
        scene, mode, space
    )
    ui.focus = FieldId(panel=transform_panel, key=key, owner=owner)
    ui.edit = begin_edit(format_transform_value(values[index]))


def commit_transform_field(
    scene: Scene,
    ui: UiState,
    mode: GizmoMode,
    space: GizmoSpace,
    keys: Sequence[str],
) -> None:
    """apply the typed value to the selection; text that is not a number is dropped"""
    if ui.focus is None or ui.edit is None:
        return
    index: int | None = field_index(keys, ui.focus.key)
    if index is None:
        return
    value: float | None = parse_transform_value(ui.edit.text)
    if value is None:
        return
    apply_transform_field(scene, mode, space, index, value)


def handle_transform_typing(
    scene: Scene,
    ui: UiState,
    mode: GizmoMode,
    space: GizmoSpace,
    owner: str,
    keys: Sequence[str],
) -> None:
    """commit on Enter, cancel on Escape, commit and step focus on Tab"""
    edit: TextEdit | None = ui.edit
    if ui.focus is None or edit is None or ui.focus.panel != transform_panel:
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
    commit_transform_field(scene, ui, mode, space, keys)
    if next_key is None:
        ui.clear_focus()
        return
    focus_transform_field(scene, ui, mode, space, owner, next_key)


def update_transform_bar(
    scene: Scene,
    ui: UiState,
    mode: GizmoMode,
    space: GizmoSpace,
    bar: TransformBarRects,
) -> None:
    """handle focus clicks and typing for the transform fields"""
    keys: tuple[str, str, str] = transform_field_keys(mode)
    sync_transform_focus(ui, bar.owner, keys)
    handle_transform_typing(scene, ui, mode, space, bar.owner, keys)

    if not is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT):
        return
    mouse: Vector2 = get_mouse_position()
    field: TransformField
    for field in bar.fields:
        if is_point_in_rect(mouse.x, mouse.y, field.rect):
            focus_transform_field(scene, ui, mode, space, bar.owner, field.key)
            return
    # only dismiss transform editing; the menu is shared with snap and other controls
    if is_point_in_rect(mouse.x, mouse.y, bar.area):
        clear_transform_focus(ui)


def draw_transform_bar(
    font: Font,
    scene: Scene,
    ui: UiState,
    mode: GizmoMode,
    space: GizmoSpace,
    bar: TransformBarRects,
) -> None:
    """draw the axis labels and field boxes"""
    size: float = float(ui_font_size)
    values: tuple[float, float, float] = transform_field_values(
        scene, mode, space
    )
    index: int
    field: TransformField
    for index, field in enumerate(bar.fields):
        draw_text_ex(
            font,
            field.label,
            Vector2(
                field.label_rect.x,
                field.label_rect.y + (field.label_rect.height - size) * 0.5,
            ),
            size,
            0,
            ui_color_text,
        )
        edit: TextEdit | None = ui.focused_edit(
            transform_panel, field.key, bar.owner
        )
        if edit is not None:
            draw_text_field(
                font,
                field.rect,
                edit.text,
                focused=True,
                caret=edit.caret,
                mark=edit.mark,
            )
            continue
        draw_text_field(
            font,
            field.rect,
            format_transform_value(values[index]),
            focused=False,
            center_unfocused=True,
        )
