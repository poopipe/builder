"""compact reusable color palette popup"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Color,
    Font,
    KeyboardKey,
    MouseButton,
    Rectangle,
    Vector2,
    draw_rectangle,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    get_mouse_position,
    is_key_pressed,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
)

from builder.ui.color_palette import default_color_palette
from builder.ui.theme import (
    ui_color_border,
    ui_color_overlay,
    ui_color_panel,
    ui_color_selection,
    ui_color_text,
    ui_font_size,
    ui_pad,
)
from builder.ui.widgets import is_point_in_rect

palette_columns: int = 8
palette_swatch_size: float = 22.0
palette_swatch_gap: float = 4.0


@dataclass(frozen=True)
class ColorPaletteCancel:
    """user cancelled the palette popup"""


@dataclass(frozen=True)
class ColorPaletteClear:
    """user cleared the color selection"""


color_palette_cancelled: ColorPaletteCancel = ColorPaletteCancel()
color_palette_cleared: ColorPaletteClear = ColorPaletteClear()

type ColorPaletteFrameResult = int | ColorPaletteClear | ColorPaletteCancel | None


@dataclass
class ColorPaletteState:
    """open palette popup targeting one owner id"""

    owner_id: str
    selected_index: int | None = None
    # optional screen anchor for the popup top-left; none centers
    anchor_x: float | None = None
    anchor_y: float | None = None
    # skip outside-press cancel until the opening click is released
    suppress_outside_cancel: bool = True


@dataclass
class ColorPaletteSwatch:
    """one laid-out palette cell"""

    index: int
    color: Color
    rect: Rectangle
    is_selected: bool = False
    is_hovered: bool = False


def make_color_palette_popup(
    *,
    owner_id: str,
    selected_index: int | None = None,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> ColorPaletteState:
    """open a palette popup for the given owner"""
    return ColorPaletteState(
        owner_id=owner_id,
        selected_index=selected_index,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        suppress_outside_cancel=True,
    )


def color_palette_window_rect(
    state: ColorPaletteState,
    screen_w: int,
    screen_h: int,
    color_count: int,
) -> Rectangle:
    """popup rect sized for an 8-wide swatch grid plus clear/title"""
    cols: int = palette_columns
    rows: int = max(1, (color_count + cols - 1) // cols)
    grid_w: float = (
        cols * palette_swatch_size + (cols - 1) * palette_swatch_gap
    )
    grid_h: float = (
        rows * palette_swatch_size + (rows - 1) * palette_swatch_gap
    )
    title_h: float = float(ui_font_size + ui_pad)
    clear_h: float = palette_swatch_size + ui_pad
    width: float = grid_w + ui_pad * 2.0
    height: float = title_h + grid_h + clear_h + ui_pad * 2.0
    x: float
    y: float
    if state.anchor_x is not None and state.anchor_y is not None:
        x = float(state.anchor_x)
        y = float(state.anchor_y)
    else:
        x = (float(screen_w) - width) * 0.5
        y = (float(screen_h) - height) * 0.5
    x = max(8.0, min(x, float(screen_w) - width - 8.0))
    y = max(8.0, min(y, float(screen_h) - height - 8.0))
    return Rectangle(x, y, width, height)


def update_color_palette_popup(
    state: ColorPaletteState,
    screen_w: int,
    screen_h: int,
    palette: tuple[Color, ...] | None = None,
) -> tuple[ColorPaletteFrameResult, list[ColorPaletteSwatch], Rectangle]:
    """handle input; return (result, swatches, window)"""
    colors: tuple[Color, ...] = (
        default_color_palette() if palette is None else palette
    )
    window: Rectangle = color_palette_window_rect(
        state, screen_w, screen_h, len(colors)
    )
    if is_key_pressed(KeyboardKey.KEY_ESCAPE):
        return color_palette_cancelled, [], window

    mouse: Vector2 = get_mouse_position()
    left_pressed: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    title_h: float = float(ui_font_size + ui_pad)
    grid_x: float = window.x + ui_pad
    grid_y: float = window.y + ui_pad + title_h
    swatches: list[ColorPaletteSwatch] = []
    result: ColorPaletteFrameResult = None
    index: int
    color: Color
    for index, color in enumerate(colors):
        col: int = index % palette_columns
        row: int = index // palette_columns
        rect: Rectangle = Rectangle(
            grid_x + col * (palette_swatch_size + palette_swatch_gap),
            grid_y + row * (palette_swatch_size + palette_swatch_gap),
            palette_swatch_size,
            palette_swatch_size,
        )
        swatch: ColorPaletteSwatch = ColorPaletteSwatch(
            index=index,
            color=color,
            rect=rect,
            is_selected=state.selected_index == index,
            is_hovered=is_point_in_rect(mouse.x, mouse.y, rect),
        )
        if swatch.is_hovered and left_released:
            result = index
        swatches.append(swatch)

    clear_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        window.y + window.height - ui_pad - palette_swatch_size,
        window.width - ui_pad * 2.0,
        palette_swatch_size,
    )
    clear_hovered: bool = is_point_in_rect(mouse.x, mouse.y, clear_rect)
    if clear_hovered and left_released:
        result = color_palette_cleared

    if state.suppress_outside_cancel:
        if not is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT):
            state.suppress_outside_cancel = False
    elif (
        left_pressed
        and not is_point_in_rect(mouse.x, mouse.y, window)
        and result is None
    ):
        result = color_palette_cancelled

    return result, swatches, window


def draw_color_palette_popup(
    font: Font,
    state: ColorPaletteState,
    swatches: list[ColorPaletteSwatch],
    window: Rectangle,
    screen_w: int,
    screen_h: int,
) -> None:
    """draw dim overlay and palette popup"""
    draw_rectangle(0, 0, screen_w, screen_h, ui_color_overlay)
    draw_rectangle_rec(window, ui_color_panel)
    draw_rectangle_lines_ex(window, 1, ui_color_border)
    draw_text_ex(
        font,
        "Color",
        Vector2(window.x + ui_pad, window.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    swatch: ColorPaletteSwatch
    for swatch in swatches:
        draw_rectangle_rec(swatch.rect, swatch.color)
        border: Color = (
            ui_color_selection if swatch.is_selected or swatch.is_hovered else ui_color_border
        )
        draw_rectangle_lines_ex(swatch.rect, 1, border)

    clear_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        window.y + window.height - ui_pad - palette_swatch_size,
        window.width - ui_pad * 2.0,
        palette_swatch_size,
    )
    draw_rectangle_rec(clear_rect, ui_color_panel)
    draw_rectangle_lines_ex(clear_rect, 1, ui_color_border)
    draw_text_ex(
        font,
        "Clear",
        Vector2(clear_rect.x + 6.0, clear_rect.y + 2.0),
        float(ui_font_size),
        0,
        ui_color_text,
    )
