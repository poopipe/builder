"""node visibility cascade, color serialize, palette load"""

from __future__ import annotations

import unittest

from pyray import Vector3

from builder.io.scene_format import node_from_json, node_to_json
from builder.scene.scene_types import Node, transform_at
from builder.scene.visibility import node_effectively_visible
from builder.ui.color_palette import (
    default_color_palette,
    load_color_palette,
    palette_color,
)


def _node(
    node_id: str,
    parent_id: str | None = None,
    *,
    visible: bool = True,
    color_index: int | None = None,
) -> Node:
    return Node(
        id=node_id,
        parent_id=parent_id,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        visible=visible,
        color_index=color_index,
    )


class TestNodeVisibility(unittest.TestCase):
    def test_own_flag_and_ancestor(self) -> None:
        nodes = {
            "root": _node("root"),
            "child": _node("child", "root"),
            "leaf": _node("leaf", "child"),
        }
        self.assertTrue(node_effectively_visible(nodes, "leaf"))
        nodes["root"] = _node("root", visible=False)
        self.assertFalse(node_effectively_visible(nodes, "leaf"))
        self.assertTrue(nodes["child"].visible)
        self.assertTrue(nodes["leaf"].visible)

    def test_child_own_hidden(self) -> None:
        nodes = {
            "root": _node("root"),
            "child": _node("child", "root", visible=False),
        }
        self.assertTrue(node_effectively_visible(nodes, "root"))
        self.assertFalse(node_effectively_visible(nodes, "child"))


class TestNodeColorSerialize(unittest.TestCase):
    def test_defaults_omitted(self) -> None:
        encoded = node_to_json(_node("a"))
        self.assertNotIn("visible", encoded)
        self.assertNotIn("color_index", encoded)

    def test_round_trip(self) -> None:
        node = _node("a", visible=False, color_index=3)
        decoded = node_from_json(node_to_json(node))
        self.assertFalse(decoded.visible)
        self.assertEqual(decoded.color_index, 3)


class TestColorPalette(unittest.TestCase):
    def test_default_has_64(self) -> None:
        palette = default_color_palette()
        self.assertEqual(len(palette), 64)
        self.assertEqual(len(load_color_palette()), 64)

    def test_palette_color_none(self) -> None:
        self.assertIsNone(palette_color(None))
        self.assertIsNone(palette_color(-1))
        self.assertIsNone(palette_color(64))
        color = palette_color(0)
        self.assertIsNotNone(color)


if __name__ == "__main__":
    unittest.main()
