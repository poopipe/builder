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
    get_time,
    is_key_down,
    is_key_pressed,
    measure_text_ex,
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
    """text being typed, with caret position and held-key repeat timing"""

    text: str
    caret: int
    select_all: bool = False
    repeat_key: int = 0
    repeat_at: float = 0.0


def begin_edit(text: str) -> TextEdit:
    """start editing with the whole value selected, so typing replaces it"""
    return TextEdit(text=text, caret=len(text), select_all=True)


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


def insert_text(edit: TextEdit, inserted: str) -> None:
    """replace the selection or insert at the caret"""
    if edit.select_all:
        edit.text = inserted
        edit.caret = len(inserted)
        edit.select_all = False
        return
    edit.text = edit.text[: edit.caret] + inserted + edit.text[edit.caret :]
    edit.caret += len(inserted)


def clear_selection(edit: TextEdit) -> None:
    """empty the field when its contents are selected"""
    edit.text = ""
    edit.caret = 0
    edit.select_all = False


def delete_before_caret(edit: TextEdit) -> None:
    """backspace"""
    if edit.select_all:
        clear_selection(edit)
        return
    if edit.caret == 0:
        return
    edit.text = edit.text[: edit.caret - 1] + edit.text[edit.caret :]
    edit.caret -= 1


def delete_at_caret(edit: TextEdit) -> None:
    """forward delete"""
    if edit.select_all:
        clear_selection(edit)
        return
    if edit.caret >= len(edit.text):
        return
    edit.text = edit.text[: edit.caret] + edit.text[edit.caret + 1 :]


def move_caret(edit: TextEdit, delta: int) -> None:
    """step the caret, collapsing a selection toward the direction of travel"""
    if edit.select_all:
        edit.select_all = False
        edit.caret = 0 if delta < 0 else len(edit.text)
        return
    edit.caret = max(0, min(len(edit.text), edit.caret + delta))


def set_caret(edit: TextEdit, caret: int) -> None:
    """place the caret and drop any selection"""
    edit.select_all = False
    edit.caret = max(0, min(len(edit.text), caret))


def shift_held() -> bool:
    """true while either shift key is down"""
    return is_key_down(KeyboardKey.KEY_LEFT_SHIFT) or is_key_down(
        KeyboardKey.KEY_RIGHT_SHIFT
    )


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

    now: float = get_time()
    if key_active(edit, KeyboardKey.KEY_LEFT, now):
        move_caret(edit, -1)
    if key_active(edit, KeyboardKey.KEY_RIGHT, now):
        move_caret(edit, 1)
    if is_key_pressed(KeyboardKey.KEY_HOME):
        set_caret(edit, 0)
    if is_key_pressed(KeyboardKey.KEY_END):
        set_caret(edit, len(edit.text))
    if key_active(edit, KeyboardKey.KEY_BACKSPACE, now):
        delete_before_caret(edit)
    if key_active(edit, KeyboardKey.KEY_DELETE, now):
        delete_at_caret(edit)

    code: int = get_char_pressed()
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
    select_all: bool = False,
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

    begin_scissor_mode(
        int(rect.x + 1.0),
        int(rect.y + 1.0),
        int(max(0.0, rect.width - 2.0)),
        int(max(0.0, rect.height - 2.0)),
    )
    if focused and select_all and text != "":
        draw_rectangle_rec(
            Rectangle(tx - 1.0, ty - 1.0, text_w + 2.0, size + 2.0),
            ui_color_selection,
        )
    draw_text_ex(font, text, Vector2(tx, ty), size, 0, ui_color_text)
    if focused and not select_all and int(get_time() * 2.0) % 2 == 0:
        draw_rectangle_rec(
            Rectangle(tx + caret_w, ty, 1.0, size),
            ui_color_text,
        )
    end_scissor_mode()
