"""heightfield albedo path scene encode/decode"""

from __future__ import annotations

import unittest
from dataclasses import replace

from builder.heightfield.heightfield_types import default_heightfield
from builder.io.scene_format import (
    heightfield_from_json,
    heightfield_to_json,
    scene_format_version,
)


class TestHeightfieldAlbedo(unittest.TestCase):
    def test_scene_format_is_37(self) -> None:
        self.assertEqual(scene_format_version, 39)

    def test_json_round_trip_null(self) -> None:
        hf = default_heightfield()
        decoded = heightfield_from_json(heightfield_to_json(hf))
        self.assertIsNone(decoded.albedo_path)

    def test_json_round_trip_path(self) -> None:
        hf = replace(default_heightfield(), albedo_path="maps/albedo.png")
        payload = heightfield_to_json(hf)
        self.assertEqual(payload["albedo_path"], "maps/albedo.png")
        decoded = heightfield_from_json(payload)
        self.assertEqual(decoded.albedo_path, "maps/albedo.png")

    def test_rejects_empty_albedo_path(self) -> None:
        payload = heightfield_to_json(default_heightfield())
        payload["albedo_path"] = ""
        with self.assertRaises(ValueError):
            heightfield_from_json(payload)


if __name__ == "__main__":
    unittest.main()
