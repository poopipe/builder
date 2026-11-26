"""per-purpose file browser directory memory and scene round-trip"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.io.scene_format import (
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    document_from_json,
    document_to_json,
    scene_format_version,
)
from builder.scene.scene_types import builtin_cube
from builder.ui.file_browser import (
    FileBrowserMode,
    FileBrowserPurpose,
    FileBrowserState,
    remember_browser_directory,
    remembered_browser_directory,
    starting_directory,
)
from pyray import Vector3


class TestRememberedBrowserDirectory(unittest.TestCase):
    def test_uses_stored_directory_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder: Path = Path(temp_dir)
            directories: dict[FileBrowserPurpose, Path] = {
                FileBrowserPurpose.import_road_shp: folder,
            }
            resolved: Path = remembered_browser_directory(
                directories,
                FileBrowserPurpose.import_road_shp,
                None,
            )
            self.assertEqual(resolved, folder.resolve())

    def test_missing_directory_falls_back_to_starting(self) -> None:
        gone: Path = Path(tempfile.gettempdir()) / "builder_missing_browser_dir"
        directories: dict[FileBrowserPurpose, Path] = {
            FileBrowserPurpose.open_scene: gone,
        }
        resolved: Path = remembered_browser_directory(
            directories,
            FileBrowserPurpose.open_scene,
            None,
        )
        self.assertEqual(resolved, starting_directory(None))

    def test_remember_stores_purpose_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder: Path = Path(temp_dir)
            directories: dict[FileBrowserPurpose, Path] = {}
            browser: FileBrowserState = FileBrowserState(
                mode=FileBrowserMode.open,
                purpose=FileBrowserPurpose.save_scene,
                directory=folder,
                filter_suffixes=(".scene",),
            )
            remember_browser_directory(directories, browser)
            self.assertEqual(
                directories[FileBrowserPurpose.save_scene], folder.resolve()
            )


class TestBrowserDirectoriesSceneRoundTrip(unittest.TestCase):
    def test_scene_format_is_37(self) -> None:
        self.assertEqual(scene_format_version, 39)

    def test_ui_browser_directories_round_trip(self) -> None:
        document: SceneDocument = SceneDocument(
            version=scene_format_version,
            active_mesh_id=builtin_cube,
            ui=SceneUiState(
                side_panel_open=True,
                meshes_panel_open=False,
                height_layers_panel_open=False,
                sculpt_panel_open=False,
                outliner_open=False,
                browser_directories={
                    "open_scene": r"D:\scenes",
                    "import_road_shp": r"D:\roads",
                },
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
        self.assertEqual(
            payload["ui"]["browser_directories"],
            {
                "open_scene": r"D:\scenes",
                "import_road_shp": r"D:\roads",
            },
        )
        decoded, warnings = document_from_json(payload)
        self.assertEqual(warnings, [])
        self.assertEqual(
            decoded.ui.browser_directories,
            {
                "open_scene": r"D:\scenes",
                "import_road_shp": r"D:\roads",
            },
        )

    def test_rejects_unknown_purpose_name(self) -> None:
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
        payload["ui"]["browser_directories"] = {"not_a_purpose": "C:/tmp"}
        with self.assertRaises(ValueError):
            document_from_json(payload)


if __name__ == "__main__":
    unittest.main()
