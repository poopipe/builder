"""corner placement uses snap/mesh origin when snaps are provided"""

from __future__ import annotations

import unittest

from builder.heightfield.road_sections.fit import (
    apply_yaw_xz,
    corner_local_centerline_xz,
    corner_snap_origin_delta_local,
    place_corner_stack_arc,
)
from builder.heightfield.road_sections.kit_widths import PieceSnapPoints
from builder.heightfield.road_sections.part_catalog import CornerPiece


class TestSnapFitOffset(unittest.TestCase):
    def test_cw_placement_puts_snap_begin_on_entry(self) -> None:
        corner = CornerPiece(angle_deg=10.0, radius_m=25.0, stem="asf1_10 25")
        # aabb-centered-ish begin/end like the real catalogue
        snaps = PieceSnapPoints(
            lb=(-6.144, 0.0, -2.692),
            pb=(5.856, 0.0, -2.692),
            le=(-5.673, 0.0, 2.692),
            pe=(6.144, 0.0, 0.608),
        )

        def snaps_for_stem(stem: str) -> PieceSnapPoints | None:
            return snaps if stem == corner.stem else None

        placed, _ex, _ez, _ey = place_corner_stack_arc(
            0.0,
            0.0,
            0.0,
            [(corner, 0.0)],
            snaps_for_stem=snaps_for_stem,
        )
        self.assertEqual(len(placed), 1)
        piece = placed[0]
        begin = snaps.begin_mid_xz()
        world = apply_yaw_xz(
            piece.x, piece.z, piece.yaw_deg, begin[0], begin[1]
        )
        self.assertAlmostEqual(world[0], 0.0, places=3)
        self.assertAlmostEqual(world[1], 0.0, places=3)

    def test_delta_matches_ideal_minus_snap(self) -> None:
        corner = CornerPiece(angle_deg=10.0, radius_m=25.0, stem="asf1_10 25")
        snaps = PieceSnapPoints(
            lb=(-6.0, 0.0, -2.7),
            pb=(6.0, 0.0, -2.7),
            le=(-6.0, 0.0, 2.7),
            pe=(6.0, 0.0, 2.7),
        )
        ideal = corner_local_centerline_xz(corner, 0.0)
        delta = corner_snap_origin_delta_local(
            corner, snaps, clockwise=True
        )
        self.assertAlmostEqual(delta[0], ideal[0] - 0.0, places=5)
        self.assertAlmostEqual(delta[1], ideal[1] - (-2.7), places=5)


if __name__ == "__main__":
    unittest.main()
