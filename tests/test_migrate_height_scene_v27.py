"""migrate_height_scene_v27 document transform"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any


def load_migrate_module() -> ModuleType:
    """load the one-off migrate script as a module"""
    path: Path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "migrate_height_scene_v27.py"
    )
    spec = importlib.util.spec_from_file_location(
        "migrate_height_scene_v27", path
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMigrateHeightSceneV27(unittest.TestCase):
    def test_adds_coverage_and_heightmap_entry(self) -> None:
        migrate = load_migrate_module()
        data: dict[str, Any] = {
            "version": 26,
            "heightmaps": [
                {
                    "id": "paint_3",
                    "label": "Road Raster",
                    "source": "heightmaps/paint_3.hraw",
                    "cache": "heightmaps/paint_3.hraw",
                    "sample_type": "f32",
                    "width": 8,
                    "height": 8,
                }
            ],
            "nodes": [
                {
                    "id": "hf",
                    "heightfield": {
                        "edit_layers": [
                            {
                                "name": "Road",
                                "height": {
                                    "source_type": "vector_road",
                                    "layer_node_id": "layer",
                                    "heightmap_id": "paint_3",
                                },
                            }
                        ]
                    },
                }
            ],
        }
        migrated: dict[str, Any] = migrate.migrate_document(data)
        self.assertEqual(migrated["version"], 27)
        height: dict[str, Any] = migrated["nodes"][0]["heightfield"][
            "edit_layers"
        ][0]["height"]
        self.assertEqual(height["heightmap_id"], "paint_3")
        coverage_id: str = height["coverage_id"]
        self.assertTrue(coverage_id.startswith("paint_"))
        self.assertNotEqual(coverage_id, "paint_3")
        ids: list[str] = [entry["id"] for entry in migrated["heightmaps"]]
        self.assertIn(coverage_id, ids)
        cov_entry: dict[str, Any] = next(
            entry
            for entry in migrated["heightmaps"]
            if entry["id"] == coverage_id
        )
        self.assertEqual(cov_entry["label"], "Road Coverage")
        self.assertEqual(cov_entry["width"], 8)
        self.assertEqual(cov_entry["height"], 8)

    def test_rejects_non_26(self) -> None:
        migrate = load_migrate_module()
        with self.assertRaises(ValueError):
            migrate.migrate_document(
                {"version": 27, "heightmaps": [], "nodes": []}
            )


if __name__ == "__main__":
    unittest.main()
