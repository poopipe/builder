"""fit centerline densify for road-sections shadows"""

from __future__ import annotations

import unittest
from math import hypot

from builder.heightfield.road_sections.fit import (
    FittedPiece,
    fitted_centerline_xz,
)
from builder.heightfield.road_sections.part_catalog import CornerPiece


class TestFittedCenterline(unittest.TestCase):
    def test_straight_spans_piece_length(self) -> None:
        pieces = [
            FittedPiece(stem="asf1_25", x=0.0, z=0.0, yaw_deg=0.0),
        ]

        def length_for_stem(stem: str) -> float:
            return 25.0 if stem.endswith("25") else 6.0

        def corner_for_stem(_stem: str) -> CornerPiece | None:
            return None

        pts = fitted_centerline_xz(
            pieces,
            length_for_stem=length_for_stem,
            corner_for_stem=corner_for_stem,
            step_m=5.0,
        )
        self.assertGreaterEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0][0], 0.0, places=5)
        self.assertAlmostEqual(pts[0][1], -12.5, places=5)
        self.assertAlmostEqual(pts[-1][0], 0.0, places=5)
        self.assertAlmostEqual(pts[-1][1], 12.5, places=5)

    def test_corner_arc_has_length(self) -> None:
        corner = CornerPiece(angle_deg=30.0, radius_m=50.0, stem="asf1_30 50")
        pieces = [
            FittedPiece(
                stem=corner.stem,
                x=0.0,
                z=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
            ),
        ]

        def length_for_stem(_stem: str) -> float:
            return 6.0

        def corner_for_stem(stem: str) -> CornerPiece | None:
            return corner if stem == corner.stem else None

        pts = fitted_centerline_xz(
            pieces,
            length_for_stem=length_for_stem,
            corner_for_stem=corner_for_stem,
            step_m=2.0,
        )
        self.assertGreaterEqual(len(pts), 3)
        chord: float = hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        self.assertGreater(chord, 1.0)
        # mid sample near the piece center
        mid = pts[len(pts) // 2]
        self.assertLess(hypot(mid[0], mid[1]), 2.0)

    def test_ccw_corner_centerline_matches_fillet(self) -> None:
        from builder.heightfield.road_sections.fit import (
            corner_placement_yaw_deg,
            place_corner_stack_arc,
        )

        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        placed, exit_x, exit_z, _exit_yaw = place_corner_stack_arc(
            0.0, 0.0, 0.0, [(corner, 180.0)]
        )
        self.assertEqual(len(placed), 1)
        self.assertAlmostEqual(placed[0].roll_deg, 0.0, places=5)
        self.assertAlmostEqual(
            placed[0].yaw_deg,
            corner_placement_yaw_deg(0.0, 90.0, clockwise=False),
            places=5,
        )
        # leading straight ending at the fillet entry so travel order is kept
        pieces = [
            FittedPiece(stem="asf1_25", x=0.0, z=-12.5, yaw_deg=0.0),
            placed[0],
        ]

        def length_for_stem(stem: str) -> float:
            return 25.0 if stem.endswith("25") else 6.0

        def corner_for_stem(stem: str) -> CornerPiece | None:
            return corner if stem == corner.stem else None

        pts = fitted_centerline_xz(
            pieces,
            length_for_stem=length_for_stem,
            corner_for_stem=corner_for_stem,
            step_m=2.0,
        )
        self.assertGreaterEqual(len(pts), 4)
        # after the approach, fillet runs entry → exit
        corner_start = next(
            i for i, p in enumerate(pts) if hypot(p[0], p[1]) < 0.05 and i > 0
        )
        self.assertAlmostEqual(pts[corner_start][0], 0.0, places=4)
        self.assertAlmostEqual(pts[corner_start][1], 0.0, places=4)
        self.assertAlmostEqual(pts[-1][0], exit_x, places=4)
        self.assertAlmostEqual(pts[-1][1], exit_z, places=4)


if __name__ == "__main__":
    unittest.main()
