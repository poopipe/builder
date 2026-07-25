"""shared text field editing: caret state, keyboard handling, and drawing"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from pyray import (
    Color,
    Font,
    KeyboardKey,
    Rectangle,
    Vector2,
    begin_scissor_mode,
    draw_rectangle_lines_ex,
    draw_rectangle_rec,
    draw_text_ex,
    end_scissor_mode,
    get_char_pressed,
    get_clipboard_text,
    get_time,
    is_key_down,
    is_key_pressed,
    measure_text_ex,
    set_clipboard_text,
)

from builder.ui.theme import (
    ui_color_border,
    ui_color_input,
    ui_color_input_focus,
    ui_color_selection,
    ui_color_text,
    ui_font_size,
)

text_repeat_delay: float = 0.35
text_repeat_interval: float = 0.04
text_field_pad_x: float = 6.0


@dataclass(frozen=True)
class FieldId:
    """identity of the focused text field"""

    panel: str
    key: str
    owner: str = ""


class TextAction(Enum):
    """what the owning panel should do after a frame of typing"""

    editing = "editing"
    commit = "commit"
    cancel = "cancel"
    focus_next = "focus_next"
    focus_prev = "focus_prev"


@dataclass
class TextEdit:
    """text being typed, with caret, optional selection mark, and key repeat"""

    text: str
    caret: int
    # selection spans mark..caret when mark is set; none means caret only
    mark: int | None = None
    repeat_key: int = 0
    repeat_at: float = 0.0


def begin_edit(text: str) -> TextEdit:
    """start editing with the whole value selected, so typing replaces it"""
    return TextEdit(text=text, caret=len(text), mark=0)


def key_active(edit: TextEdit, key: int, now: float) -> bool:
    """true on first press, then at the repeat rate while the key stays down"""
    if is_key_pressed(key):
        edit.repeat_key = int(key)
        edit.repeat_at = now + text_repeat_delay
        return True
    if edit.repeat_key != int(key) or not is_key_down(key):
        return False
    if now < edit.repeat_at:
        return False
    edit.repeat_at = now + text_repeat_interval
    return True


def selection_bounds(edit: TextEdit) -> tuple[int, int] | None:
    """return (start, end) of the selection, or none when the caret is alone"""
    if edit.mark is None or edit.mark == edit.caret:
        return None
    start: int = min(edit.mark, edit.caret)
    end: int = max(edit.mark, edit.caret)
    return start, end


def selected_text(edit: TextEdit) -> str:
    """return the selected substring, or empty when nothing is selected"""
    bounds: tuple[int, int] | None = selection_bounds(edit)
    if bounds is None:
        return ""
    start: int
    end: int
    start, end = bounds
    return edit.text[start:end]


def clear_mark(edit: TextEdit) -> None:
    """drop the selection, leaving the caret where it is"""
    edit.mark = None


def select_all(edit: TextEdit) -> None:
    """select the entire field"""
    edit.mark = 0
    edit.caret = len(edit.text)


def delete_selection(edit: TextEdit) -> bool:
    """remove the selected span; true if anything was deleted"""
    bounds: tuple[int, int] | None = selection_bounds(edit)
    if bounds is None:
        return False
    start: int
    end: int
    start, end = bounds
    edit.text = edit.text[:start] + edit.text[end:]
    edit.caret = start
    edit.mark = None
    return True


def insert_text(edit: TextEdit, inserted: str) -> None:
    """replace the selection or insert at the caret"""
    delete_selection(edit)
    edit.text = edit.text[: edit.caret] + inserted + edit.text[edit.caret :]
    edit.caret += len(inserted)
    edit.mark = None


def delete_before_caret(edit: TextEdit) -> None:
    """backspace"""
    if delete_selection(edit):
        return
    if edit.caret == 0:
        return
    edit.text = edit.text[: edit.caret - 1] + edit.text[edit.caret :]
    edit.caret -= 1


def delete_at_caret(edit: TextEdit) -> None:
    """forward delete"""
    if delete_selection(edit):
        return
    if edit.caret >= len(edit.text):
        return
    edit.text = edit.text[: edit.caret] + edit.text[edit.caret + 1 :]


def move_caret(edit: TextEdit, delta: int, *, extend: bool) -> None:
    """step the caret; extend keeps or starts a selection, otherwise clears it"""
    if not extend and selection_bounds(edit) is not None:
        start: int
        end: int
        start, end = selection_bounds(edit) or (edit.caret, edit.caret)
        edit.caret = start if delta < 0 else end
        edit.mark = None
        return
    if extend and edit.mark is None:
        edit.mark = edit.caret
    edit.caret = max(0, min(len(edit.text), edit.caret + delta))
    if not extend:
        edit.mark = None


def set_caret(edit: TextEdit, caret: int, *, extend: bool) -> None:
    """place the caret; extend keeps or starts a selection"""
    if extend and edit.mark is None:
        edit.mark = edit.caret
    edit.caret = max(0, min(len(edit.text), caret))
    if not extend:
        edit.mark = None


def shift_held() -> bool:
    """true while either shift key is down"""
    return is_key_down(KeyboardKey.KEY_LEFT_SHIFT) or is_key_down(
        KeyboardKey.KEY_RIGHT_SHIFT
    )


def ctrl_held() -> bool:
    """true while either control key is down"""
    return is_key_down(KeyboardKey.KEY_LEFT_CONTROL) or is_key_down(
        KeyboardKey.KEY_RIGHT_CONTROL
    )


def paste_clipboard(edit: TextEdit, blocked_chars: str) -> None:
    """insert clipboard text at the caret, skipping blocked characters"""
    raw: str | None = get_clipboard_text()
    if raw is None or raw == "":
        return
    cleaned: str = "".join(
        char
        for char in raw
        if char.isprintable() and char not in blocked_chars
    )
    if cleaned == "":
        return
    insert_text(edit, cleaned)


def copy_selection(edit: TextEdit) -> None:
    """copy the selection to the clipboard; no-op when nothing is selected"""
    text: str = selected_text(edit)
    if text == "":
        return
    set_clipboard_text(text)


def cut_selection(edit: TextEdit) -> None:
    """copy then delete the selection"""
    text: str = selected_text(edit)
    if text == "":
        return
    set_clipboard_text(text)
    delete_selection(edit)


def handle_text_keys(edit: TextEdit, *, blocked_chars: str = "") -> TextAction:
    """apply one frame of keyboard input; return what the owning panel should do"""
    if is_key_pressed(KeyboardKey.KEY_ESCAPE):
        return TextAction.cancel
    if is_key_pressed(KeyboardKey.KEY_ENTER) or is_key_pressed(
        KeyboardKey.KEY_KP_ENTER
    ):
        return TextAction.commit
    if is_key_pressed(KeyboardKey.KEY_TAB):
        return TextAction.focus_prev if shift_held() else TextAction.focus_next

    if ctrl_held():
        if is_key_pressed(KeyboardKey.KEY_A):
            select_all(edit)
            return TextAction.editing
        if is_key_pressed(KeyboardKey.KEY_C):
            copy_selection(edit)
            return TextAction.editing
        if is_key_pressed(KeyboardKey.KEY_X):
            cut_selection(edit)
            return TextAction.editing
        if is_key_pressed(KeyboardKey.KEY_V):
            paste_clipboard(edit, blocked_chars)
            return TextAction.editing
        # swallow other ctrl+letter keystrokes so they do not insert chars
        code: int = get_char_pressed()
        while code > 0:
            code = get_char_pressed()
        return TextAction.editing

    now: float = get_time()
    extend: bool = shift_held()
    if key_active(edit, KeyboardKey.KEY_LEFT, now):
        move_caret(edit, -1, extend=extend)
    if key_active(edit, KeyboardKey.KEY_RIGHT, now):
        move_caret(edit, 1, extend=extend)
    if is_key_pressed(KeyboardKey.KEY_HOME):
        set_caret(edit, 0, extend=extend)
    if is_key_pressed(KeyboardKey.KEY_END):
        set_caret(edit, len(edit.text), extend=extend)
    if key_active(edit, KeyboardKey.KEY_BACKSPACE, now):
        delete_before_caret(edit)
    if key_active(edit, KeyboardKey.KEY_DELETE, now):
        delete_at_caret(edit)

    code = get_char_pressed()
    while code > 0:
        char: str = chr(code)
        if char.isprintable() and char not in blocked_chars:
            insert_text(edit, char)
        code = get_char_pressed()
    return TextAction.editing


def field_after(keys: Sequence[str], key: str, step: int) -> str:
    """return the neighboring key in tab order, wrapping at the ends"""
    if not keys:
        return key
    if key not in keys:
        return keys[0]
    return keys[(keys.index(key) + step) % len(keys)]


def draw_text_field(
    font: Font,
    rect: Rectangle,
    text: str,
    *,
    focused: bool,
    caret: int = 0,
    mark: int | None = None,
    center_unfocused: bool = False,
) -> None:
    """draw an editable text box, scrolled so the caret stays visible"""
    draw_rectangle_rec(rect, ui_color_input)
    border: Color = ui_color_input_focus if focused else ui_color_border
    draw_rectangle_lines_ex(rect, 2.0 if focused else 1.0, border)

    size: float = float(ui_font_size)
    text_w: float = measure_text_ex(font, text, size, 0).x
    ty: float = rect.y + (rect.height - size) * 0.5
    inner_w: float = max(0.0, rect.width - text_field_pad_x * 2.0)

    tx: float
    if focused or not center_unfocused:
        tx = rect.x + text_field_pad_x
    else:
        tx = rect.x + (rect.width - text_w) * 0.5

    caret_index: int = max(0, min(len(text), caret))
    caret_w: float = measure_text_ex(font, text[:caret_index], size, 0).x
    if focused and caret_w > inner_w:
        tx -= caret_w - inner_w

    has_selection: bool = (
        focused and mark is not None and mark != caret and text != ""
    )

    begin_scissor_mode(
        int(rect.x + 1.0),
        int(rect.y + 1.0),
        int(max(0.0, rect.width - 2.0)),
        int(max(0.0, rect.height - 2.0)),
    )
    if has_selection:
        mark_index: int = max(0, min(len(text), mark if mark is not None else 0))
        sel_start: int = min(mark_index, caret_index)
        sel_end: int = max(mark_index, caret_index)
        sel_x0: float = tx + measure_text_ex(font, text[:sel_start], size, 0).x
        sel_x1: float = tx + measure_text_ex(font, text[:sel_end], size, 0).x
        draw_rectangle_rec(
            Rectangle(sel_x0, ty - 1.0, max(1.0, sel_x1 - sel_x0), size + 2.0),
            ui_color_selection,
        )
    draw_text_ex(font, text, Vector2(tx, ty), size, 0, ui_color_text)
    if focused and not has_selection and int(get_time() * 2.0) % 2 == 0:
        draw_rectangle_rec(
            Rectangle(tx + caret_w, ty, 1.0, size),
            ui_color_text,
        )
    end_scissor_mode()
