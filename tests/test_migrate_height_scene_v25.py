"""migrate_height_scene_v25 document transform"""

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
        / "migrate_height_scene_v25.py"
    )
    spec = importlib.util.spec_from_file_location(
        "migrate_height_scene_v25", path
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMigrateHeightSceneV25(unittest.TestCase):
    def test_renames_roles_and_adds_handles(self) -> None:
        migrate = load_migrate_module()
        data: dict[str, Any] = {
            "version": 24,
            "nodes": [
                {
                    "id": "shape",
                    "parent_id": "layer",
                    "vector_shape": {"kind": "polyline"},
                },
                {
                    "id": "p0",
                    "parent_id": "shape",
                    "role": "vector_point",
                    "vector_point": {"width_scale": 1.0},
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                },
                {
                    "id": "p1",
                    "parent_id": "shape",
                    "role": "vector_point",
                    "vector_point": {"width_scale": 2.0},
                    "transform": {
                        "translation": [10.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                },
            ],
        }
        migrated: dict[str, Any] = migrate.migrate_document(data)
        self.assertEqual(migrated["version"], 25)
        roles: set[str] = {
            str(node.get("role"))
            for node in migrated["nodes"]
            if isinstance(node, dict) and "role" in node
        }
        self.assertIn("path_point", roles)
        self.assertIn("path_handle_out", roles)
        self.assertIn("path_handle_in", roles)
        point0: dict[str, Any] = next(
            node
            for node in migrated["nodes"]
            if isinstance(node, dict) and node.get("id") == "p0"
        )
        self.assertEqual(point0.get("role"), "path_point")
        self.assertNotIn("vector_point", point0)
        self.assertEqual(point0.get("path_point"), {"width_scale": 1.0})


if __name__ == "__main__":
    unittest.main()
