"""move polylines between road usage groups"""

from __future__ import annotations

import unittest

from builder.heightfield.road_sections.edit import (
    move_polylines_between_groups,
    normalize_usage_name,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadPolyline,
    RoadUsageGroupRecipe,
)


def _line(polyline_id: str, x0: float, x1: float) -> RoadPolyline:
    return RoadPolyline(
        id=polyline_id,
        points=((x0, 0.0), (x1, 0.0)),
    )


class TestNormalizeUsageName(unittest.TestCase):
    def test_strips_and_lowercases(self) -> None:
        self.assertEqual(normalize_usage_name("  MotorWay "), "motorway")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_usage_name("   ")


class TestMovePolylinesBetweenGroups(unittest.TestCase):
    def test_moves_ids_and_clears_placements(self) -> None:
        source = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                _line("a", 0.0, 10.0),
                _line("b", 20.0, 30.0),
            ),
            placements=(
                RoadPiecePlacement(
                    polyline_id="a",
                    stem="asf1_25",
                    x=5.0,
                    z=0.0,
                    yaw_deg=0.0,
                ),
                RoadPiecePlacement(
                    polyline_id="b",
                    stem="asf1_25",
                    x=25.0,
                    z=0.0,
                    yaw_deg=0.0,
                ),
            ),
        )
        dest = RoadUsageGroupRecipe(
            usage="motorway",
            kit="asf1",
            polylines=(_line("c", 40.0, 50.0),),
        )
        new_source, new_dest = move_polylines_between_groups(
            source, dest, ["a"]
        )
        self.assertEqual([p.id for p in new_source.polylines], ["b"])
        self.assertEqual(
            [p.polyline_id for p in new_source.placements], ["b"]
        )
        self.assertEqual([p.id for p in new_dest.polylines], ["c", "a"])
        self.assertEqual(new_dest.placements, ())
        self.assertEqual(new_dest.polylines[1].points, ((0.0, 0.0), (10.0, 0.0)))

    def test_empty_selection_is_noop(self) -> None:
        source = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(_line("a", 0.0, 1.0),),
        )
        dest = RoadUsageGroupRecipe(usage="trunk", kit="asf1")
        new_source, new_dest = move_polylines_between_groups(source, dest, [])
        self.assertIs(new_source, source)
        self.assertIs(new_dest, dest)


if __name__ == "__main__":
    unittest.main()
