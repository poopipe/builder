"""terrain builder road object-import export"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pyray import Vector3

from builder.heightfield.heightfield_types import (
    default_heightfield,
    hf_local_xz_to_map_world,
    map_world_xz_to_hf_local,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadSectionsLayerRecipe,
    RoadShpRecipe,
    RoadUsageGroupRecipe,
)
from builder.tb_object import TbObject
from builder.heightfield.road_sections.tb_export import (
    export_tb_road_imports,
    mapframe_yaw_deg,
    placement_to_tb_object,
)
from builder.scene.scene_types import Node, transform_at


class TestMapframeCoords(unittest.TestCase):
    def test_local_map_roundtrip(self) -> None:
        origin_x: float = 200000.0
        origin_z: float = 5000.0
        size_x: float = 15360.0
        size_z: float = 15360.0
        world_x: float = 204456.6
        world_z: float = 4252.3
        local_x: float
        local_z: float
        local_x, local_z = map_world_xz_to_hf_local(
            world_x,
            world_z,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        back_x: float
        back_z: float
        back_x, back_z = hf_local_xz_to_map_world(
            local_x,
            local_z,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        self.assertAlmostEqual(back_x, world_x, places=6)
        self.assertAlmostEqual(back_z, world_z, places=6)

    def test_yaw_negated_for_mirror(self) -> None:
        self.assertAlmostEqual(mapframe_yaw_deg(30.0), 330.0, places=6)
        self.assertAlmostEqual(mapframe_yaw_deg(0.0), 0.0, places=6)

    def test_corner_yaw_adds_yspin_after_mirror(self) -> None:
        # -0 + (180 - 90) = 90; -(90) + (180 - 90) = 0
        self.assertAlmostEqual(
            mapframe_yaw_deg(0.0, corner_angle_deg=90.0), 90.0, places=6
        )
        self.assertAlmostEqual(
            mapframe_yaw_deg(90.0, corner_angle_deg=90.0), 0.0, places=6
        )


class TestTbRoadsExport(unittest.TestCase):
    def test_placement_line_format(self) -> None:
        placement: RoadPiecePlacement = RoadPiecePlacement(
            polyline_id="p0",
            stem="asf1_25",
            x=0.0,
            z=0.0,
            yaw_deg=30.0,
            roll_deg=5.0,
        )
        obj: TbObject = placement_to_tb_object(
            placement,
            origin_x=0.0,
            origin_z=0.0,
            size_x=256.0,
            size_z=256.0,
        )
        # center of hf → map (128, 128); yaw mirrored
        self.assertAlmostEqual(obj.x, 128.0, places=6)
        self.assertAlmostEqual(obj.y, 128.0, places=6)
        self.assertAlmostEqual(obj.yaw, 330.0, places=6)
        self.assertAlmostEqual(obj.roll, 0.0, places=6)
        self.assertEqual(
            obj.to_line(),
            '"asf1_25";128.000000;128.000000;330.000000;0.000000;0.000000;'
            "1.000000;0.000000;0;",
        )

    def test_corner_exports_yspin_roll_zero(self) -> None:
        cw: RoadPiecePlacement = RoadPiecePlacement(
            polyline_id="p0",
            stem="asf1_90 25",
            x=0.0,
            z=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
        )
        ccw: RoadPiecePlacement = RoadPiecePlacement(
            polyline_id="p0",
            stem="asf1_90 25",
            x=0.0,
            z=0.0,
            yaw_deg=90.0,
            roll_deg=0.0,
        )
        cw_obj: TbObject = placement_to_tb_object(
            cw,
            origin_x=0.0,
            origin_z=0.0,
            size_x=256.0,
            size_z=256.0,
        )
        ccw_obj: TbObject = placement_to_tb_object(
            ccw,
            origin_x=0.0,
            origin_z=0.0,
            size_x=256.0,
            size_z=256.0,
        )
        self.assertAlmostEqual(cw_obj.yaw, 90.0, places=6)
        self.assertAlmostEqual(cw_obj.roll, 0.0, places=6)
        self.assertAlmostEqual(ccw_obj.yaw, 0.0, places=6)
        self.assertAlmostEqual(ccw_obj.roll, 0.0, places=6)

    def test_corner_export_position_is_mirrored_placement(self) -> None:
        """no origin rebase: map xz is the mirrored mesh origin"""
        placement: RoadPiecePlacement = RoadPiecePlacement(
            polyline_id="p0",
            stem="asf1_10 100",
            x=12.0,
            z=-4.0,
            yaw_deg=25.0,
            roll_deg=0.0,
        )
        origin_x: float = 200000.0
        origin_z: float = 5000.0
        size_x: float = 15360.0
        size_z: float = 15360.0
        expected_x: float
        expected_z: float
        expected_x, expected_z = hf_local_xz_to_map_world(
            placement.x,
            placement.z,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        obj: TbObject = placement_to_tb_object(
            placement,
            origin_x=origin_x,
            origin_z=origin_z,
            size_x=size_x,
            size_z=size_z,
        )
        self.assertAlmostEqual(obj.x, expected_x, places=6)
        self.assertAlmostEqual(obj.y, expected_z, places=6)
        self.assertAlmostEqual(
            obj.yaw, mapframe_yaw_deg(25.0, corner_angle_deg=10.0), places=6
        )
        self.assertAlmostEqual(obj.roll, 0.0, places=6)

    def test_export_writes_per_kit_files(self) -> None:
        heightfield = replace(
            default_heightfield(),
            origin_x=replace(default_heightfield().origin_x, value=1000.0),
            origin_z=replace(default_heightfield().origin_z, value=2000.0),
            size_x=replace(default_heightfield().size_x, value=256.0),
            size_z=replace(default_heightfield().size_z, value=256.0),
        )
        identity = transform_at(Vector3(0.0, 0.0, 0.0))
        layer: Node = Node(
            id="layer",
            parent_id=None,
            transform=identity,
            mesh_id=None,
            road_sections_layer=RoadSectionsLayerRecipe(),
            name="roads",
        )
        shp: Node = Node(
            id="shp",
            parent_id="layer",
            transform=identity,
            mesh_id=None,
            road_shp=RoadShpRecipe(source_path="roads.shp"),
            name="roads.shp",
        )
        usage: Node = Node(
            id="usage",
            parent_id="shp",
            transform=identity,
            mesh_id=None,
            road_usage_group=RoadUsageGroupRecipe(
                usage="asphalt",
                kit="asf1",
                placements=(
                    RoadPiecePlacement(
                        polyline_id="line0",
                        stem="asf1_12",
                        x=10.0,
                        z=-20.0,
                        yaw_deg=90.0,
                    ),
                    RoadPiecePlacement(
                        polyline_id="line0",
                        stem="asf1_25",
                        x=-10.0,
                        z=20.0,
                        yaw_deg=0.0,
                    ),
                ),
            ),
            name="asphalt",
        )
        nodes: dict[str, Node] = {
            layer.id: layer,
            shp.id: shp,
            usage.id: usage,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out: Path = Path(temp_dir)
            result = export_tb_road_imports(nodes, layer.id, heightfield, out)
            self.assertEqual(result.objects_written, 2)
            self.assertEqual(len(result.files), 1)
            path: Path = out / "import_asf1.txt"
            self.assertTrue(path.is_file())
            lines: list[str] = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith('"asf1_12";'))
            self.assertTrue(lines[1].startswith('"asf1_25";'))


if __name__ == "__main__":
    unittest.main()
