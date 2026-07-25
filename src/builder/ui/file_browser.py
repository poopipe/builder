"""in-app modal file browser for open/save"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pyray import (
    Color,
    Font,
    KeyboardKey,
    MouseButton,
    Rectangle,
    Vector2,
    begin_scissor_mode,
    draw_rectangle,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    end_scissor_mode,
    get_char_pressed,
    get_mouse_position,
    get_mouse_wheel_move,
    get_time,
    is_key_pressed,
    is_mouse_button_down,
    is_mouse_button_pressed,
    is_mouse_button_released,
    measure_text_ex,
)

from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_color_border,
    ui_color_button,
    ui_color_button_hover,
    ui_color_button_press,
    ui_color_input,
    ui_color_input_focus,
    ui_color_overlay,
    ui_color_panel,
    ui_color_selection,
    ui_color_text,
    ui_double_click_sec,
    ui_file_browser_height,
    ui_file_browser_width,
    ui_font_size,
    ui_pad,
)
from builder.ui.widgets import Button, draw_button, is_point_in_rect, update_button

FileBrowserMode = Literal["open", "save"]
FileBrowserFrameResult = Path | Literal["cancelled"] | None


@dataclass(frozen=True)
class BrowserEntry:
    """one directory listing row"""

    name: str
    is_dir: bool
    path: Path


@dataclass
class FileBrowserState:
    """mutable modal file browser session"""

    mode: FileBrowserMode
    directory: Path
    filter_suffix: str
    filename: str = ""
    filename_focused: bool = False
    filename_select_all: bool = False
    scroll: float = 0.0
    selected_name: str | None = None
    last_click_name: str | None = None
    last_click_time: float = 0.0
    error: str = ""


@dataclass
class BrowserRow:
    """laid-out listing row for one frame"""

    entry: BrowserEntry
    rect: Rectangle = field(default_factory=lambda: Rectangle(0, 0, 0, 0))
    is_hovered: bool = False
    is_pressed: bool = False
    is_selected: bool = False


def starting_directory(scene_path: Path | None) -> Path:
    """prefer the current scene folder, else cwd"""
    if scene_path is not None:
        return scene_path.resolve().parent
    return Path.cwd().resolve()


def make_open_browser(
    directory: Path,
    filter_suffix: str,
) -> FileBrowserState:
    """open-mode browser rooted at directory"""
    return FileBrowserState(
        mode="open",
        directory=directory.resolve(),
        filter_suffix=filter_suffix.lower(),
    )


def make_save_browser(
    directory: Path,
    filter_suffix: str,
    initial_name: str,
) -> FileBrowserState:
    """save-mode browser with a suggested filename"""
    return FileBrowserState(
        mode="save",
        directory=directory.resolve(),
        filter_suffix=filter_suffix.lower(),
        filename=initial_name,
        filename_focused=True,
        filename_select_all=True,
    )


def parent_directory(path: Path) -> Path | None:
    """return parent, or None at filesystem root"""
    resolved: Path = path.resolve()
    parent: Path = resolved.parent
    if parent == resolved:
        return None
    return parent


def list_browser_entries(directory: Path, filter_suffix: str) -> list[BrowserEntry]:
    """directories first, then files matching filter_suffix"""
    entries: list[BrowserEntry] = []
    try:
        children: list[Path] = list(directory.iterdir())
    except OSError as exc:
        raise OSError(f"cannot read {directory}: {exc}") from exc
    child: Path
    for child in children:
        try:
            is_dir: bool = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append(BrowserEntry(name=child.name, is_dir=True, path=child))
            continue
        if child.suffix.lower() == filter_suffix:
            entries.append(BrowserEntry(name=child.name, is_dir=False, path=child))
    entries.sort(key=lambda item: (not item.is_dir, item.name.lower()))
    return entries


def ensure_suffix(path: Path, suffix: str) -> Path:
    """append suffix when missing"""
    if path.suffix.lower() == suffix.lower():
        return path
    return path.with_suffix(suffix)


def browser_window_rect(screen_w: int, screen_h: int) -> Rectangle:
    """centered dialog rect"""
    width: float = min(ui_file_browser_width, float(screen_w) - 40.0)
    height: float = min(ui_file_browser_height, float(screen_h) - 40.0)
    return Rectangle(
        (float(screen_w) - width) * 0.5,
        (float(screen_h) - height) * 0.5,
        width,
        height,
    )


def navigate_to(state: FileBrowserState, directory: Path) -> None:
    """enter a directory and reset list selection"""
    state.directory = directory.resolve()
    state.scroll = 0.0
    state.selected_name = None
    state.last_click_name = None
    state.error = ""


def resolve_confirm_path(state: FileBrowserState) -> Path | None:
    """path for Open/Save confirm, or None if invalid"""
    if state.mode == "open":
        if state.selected_name is None:
            state.error = "select a file"
            return None
        path: Path = state.directory / state.selected_name
        if not path.is_file():
            state.error = "select a file"
            return None
        if path.suffix.lower() != state.filter_suffix:
            state.error = f"choose a {state.filter_suffix} file"
            return None
        return path.resolve()
    name: str = state.filename.strip()
    if name == "":
        state.error = "enter a file name"
        return None
    return ensure_suffix(state.directory / name, state.filter_suffix).resolve()


def handle_filename_typing(state: FileBrowserState) -> None:
    """edit save filename while focused"""
    if not state.filename_focused:
        return
    if is_key_pressed(KeyboardKey.KEY_ESCAPE):
        state.filename_focused = False
        state.filename_select_all = False
        return
    if is_key_pressed(KeyboardKey.KEY_BACKSPACE):
        if state.filename_select_all:
            state.filename = ""
            state.filename_select_all = False
        else:
            state.filename = state.filename[:-1]
        return
    code: int = get_char_pressed()
    while code > 0:
        char: str = chr(code)
        if char.isprintable() and char not in ('"', "*", "?", "<", ">", "|"):
            if state.filename_select_all:
                state.filename = char
                state.filename_select_all = False
            else:
                state.filename += char
        code = get_char_pressed()


def activate_entry(state: FileBrowserState, entry: BrowserEntry) -> FileBrowserFrameResult:
    """navigate dirs; select/open files"""
    if entry.is_dir:
        navigate_to(state, entry.path)
        return None
    state.selected_name = entry.name
    state.error = ""
    if state.mode == "save":
        state.filename = entry.name
        state.filename_focused = False
        state.filename_select_all = False
        return None
    now: float = get_time()
    is_double: bool = (
        state.last_click_name == entry.name
        and (now - state.last_click_time) <= ui_double_click_sec
    )
    state.last_click_name = entry.name
    state.last_click_time = now
    if is_double:
        chosen: Path | None = resolve_confirm_path(state)
        return chosen
    return None


def update_file_browser(
    state: FileBrowserState,
    screen_w: int,
    screen_h: int,
) -> tuple[FileBrowserFrameResult, list[BrowserRow], list[Button], Rectangle]:
    """handle input; return (result, rows, buttons, window)"""
    window: Rectangle = browser_window_rect(screen_w, screen_h)
    title_h: float = float(ui_font_size + ui_pad)
    path_h: float = float(ui_font_size)
    footer_h: float = float(ui_button_height + ui_pad * 2)
    name_h: float = (
        float(ui_button_height + ui_pad) if state.mode == "save" else 0.0
    )
    list_top: float = window.y + ui_pad + title_h + path_h + ui_pad
    list_h: float = max(
        0.0,
        window.height
        - ui_pad * 2.0
        - title_h
        - path_h
        - ui_pad
        - name_h
        - footer_h,
    )
    list_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        list_top,
        window.width - ui_pad * 2.0,
        list_h,
    )
    name_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        list_top + list_h + ui_pad,
        window.width - ui_pad * 2.0,
        float(ui_button_height),
    )
    btn_y: float = (
        window.y + window.height - ui_pad - float(ui_button_height)
    )
    btn_w: float = 88.0
    cancel_rect: Rectangle = Rectangle(
        window.x + window.width - ui_pad - btn_w,
        btn_y,
        btn_w,
        float(ui_button_height),
    )
    confirm_rect: Rectangle = Rectangle(
        cancel_rect.x - ui_button_gap - btn_w,
        btn_y,
        btn_w,
        float(ui_button_height),
    )
    up_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        btn_y,
        72.0,
        float(ui_button_height),
    )

    result: FileBrowserFrameResult = None
    if is_key_pressed(KeyboardKey.KEY_ESCAPE) and not state.filename_focused:
        return "cancelled", [], [], window

    handle_filename_typing(state)

    entries: list[BrowserEntry] = []
    try:
        entries = list_browser_entries(state.directory, state.filter_suffix)
    except OSError as exc:
        state.error = str(exc)

    row_h: float = float(ui_button_height)
    content_h: float = max(
        0.0, len(entries) * (row_h + ui_button_gap) - ui_button_gap
    )
    max_scroll: float = max(0.0, content_h - list_h)
    mouse: Vector2 = get_mouse_position()
    if is_point_in_rect(mouse.x, mouse.y, list_rect):
        wheel: float = get_mouse_wheel_move()
        if wheel != 0.0:
            state.scroll -= wheel * row_h
    state.scroll = max(0.0, min(max_scroll, state.scroll))

    left_down: bool = is_mouse_button_down(MouseButton.MOUSE_BUTTON_LEFT)
    left_pressed: bool = is_mouse_button_pressed(MouseButton.MOUSE_BUTTON_LEFT)
    left_released: bool = is_mouse_button_released(MouseButton.MOUSE_BUTTON_LEFT)
    pointer_in_list: bool = is_point_in_rect(mouse.x, mouse.y, list_rect)

    rows: list[BrowserRow] = []
    y: float = list_top - state.scroll
    entry: BrowserEntry
    for entry in entries:
        row: BrowserRow = BrowserRow(
            entry=entry,
            rect=Rectangle(list_rect.x, y, list_rect.width, row_h),
            is_selected=state.selected_name == entry.name,
        )
        row.is_hovered = is_point_in_rect(mouse.x, mouse.y, row.rect)
        row.is_pressed = row.is_hovered and left_down
        visible: bool = (
            row.rect.y + row.rect.height > list_rect.y
            and row.rect.y < list_rect.y + list_rect.height
        )
        if pointer_in_list and visible and row.is_hovered and left_released:
            activated: FileBrowserFrameResult = activate_entry(state, entry)
            if activated is not None:
                result = activated
        rows.append(row)
        y += row_h + ui_button_gap

    if state.mode == "save" and left_pressed:
        if is_point_in_rect(mouse.x, mouse.y, name_rect):
            state.filename_focused = True
            state.filename_select_all = True
        elif not is_point_in_rect(mouse.x, mouse.y, confirm_rect):
            state.filename_focused = False
            state.filename_select_all = False

    def confirm() -> None:
        nonlocal result
        chosen: Path | None = resolve_confirm_path(state)
        if chosen is not None:
            result = chosen

    def cancel() -> None:
        nonlocal result
        result = "cancelled"

    def go_up() -> None:
        parent: Path | None = parent_directory(state.directory)
        if parent is None:
            state.error = "already at drive root"
            return
        navigate_to(state, parent)

    confirm_label: str = "Open" if state.mode == "open" else "Save"
    buttons: list[Button] = [
        Button("Up", go_up, up_rect),
        Button(confirm_label, confirm, confirm_rect),
        Button("Cancel", cancel, cancel_rect),
    ]
    button: Button
    for button in buttons:
        update_button(button, mouse, left_down, left_released)

    if state.mode == "save" and state.filename_focused:
        if is_key_pressed(KeyboardKey.KEY_ENTER) or is_key_pressed(
            KeyboardKey.KEY_KP_ENTER
        ):
            confirm()

    return result, rows, buttons, window


def draw_browser_row(font: Font, row: BrowserRow) -> None:
    """draw one directory listing row"""
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
    prefix: str = "[dir] " if row.entry.is_dir else "      "
    label: str = prefix + row.entry.name
    draw_text_ex(
        font,
        label,
        Vector2(row.rect.x + 8.0, row.rect.y + 4.0),
        float(ui_font_size),
        0,
        ui_color_text,
    )


def draw_file_browser(
    font: Font,
    state: FileBrowserState,
    rows: list[BrowserRow],
    buttons: list[Button],
    window: Rectangle,
    screen_w: int,
    screen_h: int,
) -> None:
    """draw dim overlay and browser dialog"""
    draw_rectangle(0, 0, screen_w, screen_h, ui_color_overlay)
    draw_rectangle_rec(window, ui_color_panel)
    draw_rectangle_lines_ex(window, 1, ui_color_border)

    title: str = "Open scene" if state.mode == "open" else "Save scene"
    draw_text_ex(
        font,
        title,
        Vector2(window.x + ui_pad, window.y + ui_pad),
        float(ui_font_size),
        0,
        ui_color_text,
    )
    path_text: str = str(state.directory)
    path_size: float = float(ui_font_size - 2)
    max_path_w: float = window.width - ui_pad * 2.0
    if measure_text_ex(font, path_text, path_size, 0).x > max_path_w:
        while len(path_text) > 4 and measure_text_ex(
            font, "..." + path_text, path_size, 0
        ).x > max_path_w:
            path_text = path_text[1:]
        path_text = "..." + path_text
    draw_text_ex(
        font,
        path_text,
        Vector2(
            window.x + ui_pad,
            window.y + ui_pad + float(ui_font_size),
        ),
        path_size,
        0,
        ui_color_text,
    )

    title_h: float = float(ui_font_size + ui_pad)
    path_h: float = float(ui_font_size)
    footer_h: float = float(ui_button_height + ui_pad * 2)
    name_h: float = (
        float(ui_button_height + ui_pad) if state.mode == "save" else 0.0
    )
    list_top: float = window.y + ui_pad + title_h + path_h + ui_pad
    list_h: float = max(
        0.0,
        window.height
        - ui_pad * 2.0
        - title_h
        - path_h
        - ui_pad
        - name_h
        - footer_h,
    )
    list_rect: Rectangle = Rectangle(
        window.x + ui_pad,
        list_top,
        window.width - ui_pad * 2.0,
        list_h,
    )
    begin_scissor_mode(
        int(list_rect.x),
        int(list_rect.y),
        int(list_rect.width),
        int(list_rect.height),
    )
    row: BrowserRow
    for row in rows:
        draw_browser_row(font, row)
    end_scissor_mode()
    draw_rectangle_lines_ex(list_rect, 1, ui_color_border)

    if state.mode == "save":
        name_rect: Rectangle = Rectangle(
            window.x + ui_pad,
            list_top + list_h + ui_pad,
            window.width - ui_pad * 2.0,
            float(ui_button_height),
        )
        draw_rectangle_rec(name_rect, ui_color_input)
        border: Color = (
            ui_color_input_focus if state.filename_focused else ui_color_border
        )
        draw_rectangle_lines_ex(
            name_rect,
            2.0 if state.filename_focused else 1.0,
            border,
        )
        text: str = state.filename
        tx: float = name_rect.x + 8.0
        ty: float = (
            name_rect.y + (name_rect.height - float(ui_font_size)) * 0.5
        )
        if state.filename_focused and state.filename_select_all and text != "":
            text_w: float = measure_text_ex(
                font, text, float(ui_font_size), 0
            ).x
            draw_rectangle_rec(
                Rectangle(tx, ty, text_w, float(ui_font_size)),
                ui_color_selection,
            )
        draw_text_ex(
            font,
            text,
            Vector2(tx, ty),
            float(ui_font_size),
            0,
            ui_color_text,
        )

    if state.error != "":
        draw_text_ex(
            font,
            state.error,
            Vector2(
                window.x + ui_pad + 80.0,
                window.y + window.height - ui_pad - 22.0,
            ),
            float(ui_font_size - 4),
            0,
            ui_color_text,
        )

    button: Button
    for button in buttons:
        draw_button(button, font)
