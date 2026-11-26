"""tabbed docks layout, tab fallback, and scene persistence"""

from __future__ import annotations

import unittest

from builder.io.scene_format import (
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    document_from_json,
    document_to_json,
    scene_format_version,
)
from builder.scene.scene_types import builtin_cube
from builder.ui.dock import (
    DockPanel,
    RightDockContent,
    available_left_tabs,
    available_right_tabs,
    clamp_dock_width,
    resolve_active_tab,
)
from builder.ui.theme import ui_dock_max_width, ui_dock_min_width, ui_tab_bar_height
from builder.ui.widgets import compute_layout
from pyray import Vector3


class TestDockLayout(unittest.TestCase):
    def test_two_docks_and_viewport(self) -> None:
        layout = compute_layout(1000, 600, left_width=200.0, right_width=220.0)
        self.assertEqual(layout.left_dock.width, 200.0)
        self.assertEqual(layout.right_dock.width, 220.0)
        self.assertEqual(layout.viewport.width, 580.0)
        self.assertEqual(layout.left_content.height, 600 - 36 - 24 - ui_tab_bar_height)
        self.assertEqual(layout.left_content.y, layout.left_dock.y + ui_tab_bar_height)

    def test_zero_width_collapses_dock(self) -> None:
        layout = compute_layout(800, 400, left_width=0.0, right_width=0.0)
        self.assertEqual(layout.left_dock.width, 0.0)
        self.assertEqual(layout.right_dock.width, 0.0)
        self.assertEqual(layout.viewport.width, 800.0)
        self.assertEqual(layout.left_content.height, 0.0)


class TestDockTabs(unittest.TestCase):
    def test_available_left_tabs(self) -> None:
        self.assertEqual(
            available_left_tabs(side_panel_open=True, outliner_open=False),
            [DockPanel.tools],
        )
        self.assertEqual(
            available_left_tabs(side_panel_open=True, outliner_open=True),
            [DockPanel.tools, DockPanel.outliner],
        )

    def test_available_right_tabs_order(self) -> None:
        tabs = available_right_tabs(
            meshes_panel_open=True,
            height_layers_panel_open=True,
            sculpt_panel_open=False,
            content=RightDockContent(show_inspector=True, show_modifiers=True),
        )
        self.assertEqual(
            tabs,
            [
                DockPanel.inspector,
                DockPanel.modifiers,
                DockPanel.meshes,
                DockPanel.height_layers,
            ],
        )

    def test_resolve_keeps_current(self) -> None:
        available = [DockPanel.meshes, DockPanel.sculpt]
        self.assertEqual(
            resolve_active_tab(DockPanel.sculpt, available, DockPanel.inspector),
            DockPanel.sculpt,
        )

    def test_resolve_prefers_when_missing(self) -> None:
        available = [DockPanel.inspector, DockPanel.meshes]
        self.assertEqual(
            resolve_active_tab(DockPanel.sculpt, available, DockPanel.inspector),
            DockPanel.inspector,
        )

    def test_resolve_empty(self) -> None:
        self.assertIsNone(resolve_active_tab(DockPanel.meshes, [], DockPanel.inspector))

    def test_clamp_dock_width(self) -> None:
        self.assertEqual(clamp_dock_width(10.0), ui_dock_min_width)
        self.assertEqual(clamp_dock_width(9999.0), ui_dock_max_width)


class TestDockSceneRoundTrip(unittest.TestCase):
    def test_scene_format_is_37(self) -> None:
        self.assertEqual(scene_format_version, 39)

    def test_dock_fields_round_trip(self) -> None:
        document: SceneDocument = SceneDocument(
            version=scene_format_version,
            active_mesh_id=builtin_cube,
            ui=SceneUiState(
                side_panel_open=True,
                meshes_panel_open=True,
                height_layers_panel_open=False,
                sculpt_panel_open=False,
                outliner_open=True,
                left_dock_width=180.0,
                right_dock_width=300.0,
                left_dock_tab=DockPanel.tools.name,
                right_dock_tab=DockPanel.height_layers.name,
            ),
            camera=SceneCameraState(
                yaw=0.0,
                pitch=0.0,
                distance=10.0,
                target=Vector3(0.0, 0.0, 0.0),
                zoom_speed=1.0,
                move_speed=1.0,
                view_distance=1000.0,
            ),
            meshes=(),
            heightmaps=(),
            nodes=(),
        )
        payload = document_to_json(document, {}, {}, {})
        self.assertEqual(payload["ui"]["left_dock_width"], 180.0)
        self.assertEqual(payload["ui"]["right_dock_width"], 300.0)
        self.assertEqual(payload["ui"]["left_dock_tab"], "tools")
        self.assertEqual(payload["ui"]["right_dock_tab"], "height_layers")
        decoded, warnings = document_from_json(payload)
        self.assertEqual(warnings, [])
        self.assertEqual(decoded.ui.left_dock_width, 180.0)
        self.assertEqual(decoded.ui.right_dock_width, 300.0)
        self.assertEqual(decoded.ui.left_dock_tab, "tools")
        self.assertEqual(decoded.ui.right_dock_tab, "height_layers")

    def test_rejects_unknown_tab(self) -> None:
        document: SceneDocument = SceneDocument(
            version=scene_format_version,
            active_mesh_id=builtin_cube,
            ui=SceneUiState(
                side_panel_open=True,
                meshes_panel_open=False,
                height_layers_panel_open=False,
                sculpt_panel_open=False,
                outliner_open=False,
            ),
            camera=SceneCameraState(
                yaw=0.0,
                pitch=0.0,
                distance=10.0,
                target=Vector3(0.0, 0.0, 0.0),
                zoom_speed=1.0,
                move_speed=1.0,
                view_distance=1000.0,
            ),
            meshes=(),
            heightmaps=(),
            nodes=(),
        )
        payload = document_to_json(document, {}, {}, {})
        payload["ui"]["left_dock_tab"] = "not_a_panel"
        with self.assertRaises(ValueError):
            document_from_json(payload)

    def test_rejects_non_numeric_width(self) -> None:
        document: SceneDocument = SceneDocument(
            version=scene_format_version,
            active_mesh_id=builtin_cube,
            ui=SceneUiState(
                side_panel_open=True,
                meshes_panel_open=False,
                height_layers_panel_open=False,
                sculpt_panel_open=False,
                outliner_open=False,
            ),
            camera=SceneCameraState(
                yaw=0.0,
                pitch=0.0,
                distance=10.0,
                target=Vector3(0.0, 0.0, 0.0),
                zoom_speed=1.0,
                move_speed=1.0,
                view_distance=1000.0,
            ),
            meshes=(),
            heightmaps=(),
            nodes=(),
        )
        payload = document_to_json(document, {}, {}, {})
        payload["ui"]["right_dock_width"] = "wide"
        with self.assertRaises(ValueError):
            document_from_json(payload)


if __name__ == "__main__":
    unittest.main()
