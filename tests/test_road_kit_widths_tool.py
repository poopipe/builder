"""unit tests for tools.build_road_kit_widths helpers"""

from __future__ import annotations

import unittest

from tools.build_road_kit_widths import (
    PieceWidthRow,
    aggregate_kits,
    width_from_straight_sizes,
)


class TestRoadKitWidthsTool(unittest.TestCase):
    def test_width_from_straight_sizes_z_is_length(self) -> None:
        width, measured = width_from_straight_sizes((10.0, 0.2, 25.0), 25.0)
        self.assertAlmostEqual(width, 10.0)
        self.assertAlmostEqual(measured, 25.0)

    def test_width_from_straight_sizes_x_is_length(self) -> None:
        width, measured = width_from_straight_sizes((25.0, 0.2, 10.0), 25.0)
        self.assertAlmostEqual(width, 10.0)
        self.assertAlmostEqual(measured, 25.0)

    def test_aggregate_kits_median(self) -> None:
        rows = [
            PieceWidthRow(
                kit="asf1",
                stem="asf1_25",
                piece_type="straight",
                path="a.p3d",
                length_m=25.0,
                size_x=10.0,
                size_y=0.2,
                size_z=25.0,
                width_m=10.0,
                length_axis_m=25.0,
                aabb_source="odol_modelinfo",
            ),
            PieceWidthRow(
                kit="asf1",
                stem="asf1_12",
                piece_type="straight",
                path="b.p3d",
                length_m=12.0,
                size_x=10.2,
                size_y=0.2,
                size_z=12.0,
                width_m=10.2,
                length_axis_m=12.0,
                aabb_source="odol_modelinfo",
            ),
            PieceWidthRow(
                kit="asf1",
                stem="asf1_10",
                piece_type="straight",
                path="c.p3d",
                length_m=10.0,
                size_x=9.8,
                size_y=0.2,
                size_z=10.0,
                width_m=9.8,
                length_axis_m=10.0,
                aabb_source="odol_modelinfo",
            ),
            PieceWidthRow(
                kit="asf1",
                stem="asf1_100 25",
                piece_type="corner",
                path="d.p3d",
                length_m=None,
                size_x=20.0,
                size_y=0.2,
                size_z=20.0,
                width_m=None,
                length_axis_m=None,
                aabb_source="odol_modelinfo",
            ),
        ]
        kits = aggregate_kits(rows)
        self.assertEqual(len(kits), 1)
        self.assertEqual(kits[0].kit, "asf1")
        self.assertEqual(kits[0].straight_count, 3)
        self.assertIsNotNone(kits[0].width_m)
        assert kits[0].width_m is not None
        self.assertAlmostEqual(kits[0].width_m, 10.0)


if __name__ == "__main__":
    unittest.main()
