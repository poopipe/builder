"""list picker state and cancel paths"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pyray import KeyboardKey, Vector2

from builder.ui.list_picker import (
    ListPickerCancel,
    ListPickerPurpose,
    make_list_picker,
    update_list_picker,
)


class TestMakeListPicker(unittest.TestCase):
    def test_clamps_selected_into_range(self) -> None:
        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=("a", "b", "c"),
            selected=99,
            target_id="node-1",
        )
        self.assertEqual(state.selected, 2)
        self.assertEqual(state.target_id, "node-1")
        self.assertEqual(state.items, ("a", "b", "c"))

    def test_negative_selected_clamps_to_zero(self) -> None:
        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=("a", "b"),
            selected=-3,
        )
        self.assertEqual(state.selected, 0)

    def test_empty_items_selected_stays_zero(self) -> None:
        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=(),
            selected=5,
        )
        self.assertEqual(state.items, ())
        self.assertEqual(state.selected, 0)


class TestUpdateListPicker(unittest.TestCase):
    def test_escape_cancels(self) -> None:
        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=("asf1", "asf2"),
            selected=1,
        )

        def key_pressed(key: KeyboardKey) -> bool:
            return key == KeyboardKey.KEY_ESCAPE

        with patch("builder.ui.list_picker.is_key_pressed", side_effect=key_pressed):
            result, rows, buttons, window = update_list_picker(state, 1280, 720)
        self.assertIs(result, ListPickerCancel.cancelled)
        self.assertEqual(rows, [])
        self.assertEqual(buttons, [])
        self.assertGreater(window.width, 0.0)
        self.assertGreater(window.height, 0.0)

    def test_empty_list_escape_still_cancels(self) -> None:
        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=(),
        )

        def key_pressed(key: KeyboardKey) -> bool:
            return key == KeyboardKey.KEY_ESCAPE

        with patch("builder.ui.list_picker.is_key_pressed", side_effect=key_pressed):
            result, _rows, _buttons, _window = update_list_picker(state, 800, 600)
        self.assertIs(result, ListPickerCancel.cancelled)

    def test_ok_confirms_selected(self) -> None:
        from builder.ui.theme import (
            ui_button_gap,
            ui_button_height,
            ui_file_browser_height,
            ui_file_browser_width,
            ui_pad,
        )

        state = make_list_picker(
            purpose=ListPickerPurpose.road_kit,
            title="Select kit",
            items=("kit_a", "kit_b", "kit_c"),
            selected=1,
        )

        def key_pressed(_key: KeyboardKey) -> bool:
            return False

        screen_w: int = 1280
        screen_h: int = 720
        window_w: float = min(ui_file_browser_width * 0.7, float(screen_w) - 40.0)
        window_h: float = min(ui_file_browser_height * 0.85, float(screen_h) - 40.0)
        window_x: float = (float(screen_w) - window_w) * 0.5
        window_y: float = (float(screen_h) - window_h) * 0.5
        btn_w: float = 88.0
        cancel_x: float = window_x + window_w - ui_pad - btn_w
        confirm_x: float = cancel_x - ui_button_gap - btn_w
        btn_y: float = window_y + window_h - ui_pad - float(ui_button_height)
        mouse: Vector2 = Vector2(
            confirm_x + btn_w * 0.5,
            btn_y + float(ui_button_height) * 0.5,
        )

        with (
            patch("builder.ui.list_picker.is_key_pressed", side_effect=key_pressed),
            patch(
                "builder.ui.list_picker.get_mouse_position",
                return_value=mouse,
            ),
            patch("builder.ui.list_picker.get_mouse_wheel_move", return_value=0.0),
            patch(
                "builder.ui.list_picker.is_mouse_button_down",
                return_value=False,
            ),
            patch(
                "builder.ui.list_picker.is_mouse_button_released",
                return_value=True,
            ),
        ):
            result, _rows, _buttons, _window = update_list_picker(
                state, screen_w, screen_h
            )
        self.assertEqual(result, "kit_b")


if __name__ == "__main__":
    unittest.main()
