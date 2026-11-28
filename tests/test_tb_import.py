"""tb object-import parse and mapframe placement"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.generators.tb_import import (
    local_yaw_from_mapframe,
    transform_from_tb_object,
)
from builder.heightfield.heightfield_types import map_world_xz_to_hf_local
from builder.heightfield.road_sections.tb_export import mapframe_yaw_deg
from builder.tb_object import (
    TbObject,
    parse_tb_object_import,
    parse_tb_object_line,
)


class TestParseTbObject(unittest.TestCase):
    def test_parse_line(self) -> None:
        obj: TbObject | None = parse_tb_object_line(
            '"tenement_big";207174.598;7738.938;209.193;0.000;0.000;1.000;2.331;0;'
        )
        assert obj is not None
        self.assertEqual(obj.model, "tenement_big")
        self.assertAlmostEqual(obj.x, 207174.598)
        self.assertAlmostEqual(obj.y, 7738.938)
        self.assertAlmostEqual(obj.yaw, 209.193)
        self.assertAlmostEqual(obj.scale, 1.0)
        self.assertEqual(obj.underground, 0)

    def test_parse_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path: Path = Path(tmp) / "import_houses.txt"
            path.write_text(
                '"tenement_big";100.0;200.0;90.0;0;0;1;0;0;\n'
                "\n"
                '"tenement_small";110.0;210.0;45.0;0;0;1.5;0;0;\n',
                encoding="utf-8",
            )
            objects: list[TbObject] = parse_tb_object_import(path)
            self.assertEqual(len(objects), 2)
            self.assertEqual(objects[1].model, "tenement_small")
            self.assertAlmostEqual(objects[1].scale, 1.5)


class TestTbImportFrame(unittest.TestCase):
    def test_yaw_roundtrip_sign(self) -> None:
        self.assertAlmostEqual(local_yaw_from_mapframe(90.0), -90.0)
        self.assertAlmostEqual(
            mapframe_yaw_deg(local_yaw_from_mapframe(90.0)), 90.0
        )

    def test_transform_uses_hf_local_xz(self) -> None:
        origin_x: float = 200000.0
        origin_z: float = 5000.0
        size_x: float = 256.0
        size_z: float = 256.0
        world_x: float = 200100.0
        world_z: float = 5100.0
        expected_x: float
        expected_z: float
        expected_x, expected_z = map_world_xz_to_hf_local(
            world_x,
            world_z,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        transform = transform_from_tb_object(
            TbObject(model="box", x=world_x, y=world_z, yaw=90.0, scale=2.0),
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        self.assertAlmostEqual(transform.translation.x, expected_x)
        self.assertAlmostEqual(transform.translation.y, 0.0)
        self.assertAlmostEqual(transform.translation.z, expected_z)
        self.assertAlmostEqual(transform.scale.x, 2.0)


if __name__ == "__main__":
    unittest.main()
