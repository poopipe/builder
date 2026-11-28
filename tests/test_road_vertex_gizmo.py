"""road polyline vertex handles and xz move gizmo"""

from __future__ import annotations

import unittest

from pyray import Matrix, Vector3, matrix_identity

from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPolyline,
    RoadUsageGroupRecipe,
)
from builder.heightfield.road_sections.vertex_gizmo import (
    EndpointStart,
    apply_endpoint_starts_delta,
    selection_centroid_xz,
    world_delta_to_hf_xz,
)
from builder.heightfield.road_sections.vertex_handle import (
    vertex_cube_base_size,
    vertex_cube_draw_size,
    vertex_pick_half_extent,
    vertex_pick_radius,
)


def _group(*polylines: RoadPolyline) -> RoadUsageGroupRecipe:
    return RoadUsageGroupRecipe(
        usage="secondary",
        kit="asf3",
        polylines=polylines,
    )


class TestVertexHandleSize(unittest.TestCase):
    def test_pick_floor_matches_half_cube(self) -> None:
        self.assertAlmostEqual(
            vertex_pick_half_extent, vertex_cube_base_size * 0.5
        )

    def test_draw_size_at_least_base(self) -> None:
        self.assertAlmostEqual(vertex_cube_draw_size(0.0), vertex_cube_base_size)
        self.assertGreater(vertex_cube_draw_size(2.0), vertex_cube_base_size)

    def test_pick_radius_at_least_half_extent(self) -> None:
        self.assertGreaterEqual(vertex_pick_radius(0.0), vertex_pick_half_extent)


class TestSelectionCentroid(unittest.TestCase):
    def test_single_endpoint(self) -> None:
        line = RoadPolyline(
            id="a",
            points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        )
        group = _group(line)
        xz = selection_centroid_xz(
            group, [RoadEndpointRef(polyline_id="a", vertex_index=1)]
        )
        self.assertEqual(xz, (10.0, 0.0))

    def test_two_endpoints_midpoint(self) -> None:
        line = RoadPolyline(
            id="a",
            points=((0.0, 0.0), (10.0, 0.0), (10.0, 20.0)),
        )
        group = _group(line)
        xz = selection_centroid_xz(
            group,
            [
                RoadEndpointRef(polyline_id="a", vertex_index=0),
                RoadEndpointRef(polyline_id="a", vertex_index=2),
            ],
        )
        self.assertIsNotNone(xz)
        assert xz is not None
        self.assertAlmostEqual(xz[0], 5.0)
        self.assertAlmostEqual(xz[1], 10.0)

    def test_empty_selection(self) -> None:
        group = _group(RoadPolyline(id="a", points=((0.0, 0.0), (1.0, 1.0))))
        self.assertIsNone(selection_centroid_xz(group, []))


class TestApplyEndpointDelta(unittest.TestCase):
    def test_moves_one_endpoint(self) -> None:
        line = RoadPolyline(
            id="a",
            points=((0.0, 0.0), (10.0, 0.0)),
        )
        group = _group(line)
        updated = apply_endpoint_starts_delta(
            group,
            [
                EndpointStart(
                    endpoint=RoadEndpointRef(polyline_id="a", vertex_index=0),
                    x=0.0,
                    z=0.0,
                )
            ],
            3.0,
            -2.0,
        )
        self.assertEqual(updated.polylines[0].points[0], (3.0, -2.0))
        self.assertEqual(updated.polylines[0].points[1], (10.0, 0.0))

    def test_moves_two_endpoints_same_delta(self) -> None:
        line_a = RoadPolyline(
            id="a",
            points=((0.0, 0.0), (10.0, 0.0)),
        )
        line_b = RoadPolyline(
            id="b",
            points=((5.0, 5.0), (15.0, 5.0)),
        )
        group = _group(line_a, line_b)
        updated = apply_endpoint_starts_delta(
            group,
            [
                EndpointStart(
                    endpoint=RoadEndpointRef(polyline_id="a", vertex_index=0),
                    x=0.0,
                    z=0.0,
                ),
                EndpointStart(
                    endpoint=RoadEndpointRef(polyline_id="b", vertex_index=1),
                    x=15.0,
                    z=5.0,
                ),
            ],
            2.0,
            4.0,
        )
        self.assertEqual(updated.polylines[0].points[0], (2.0, 4.0))
        self.assertEqual(updated.polylines[0].points[1], (10.0, 0.0))
        self.assertEqual(updated.polylines[1].points[0], (5.0, 5.0))
        self.assertEqual(updated.polylines[1].points[1], (17.0, 9.0))


class TestWorldDeltaToHfXz(unittest.TestCase):
    def test_identity_keeps_xz(self) -> None:
        identity: Matrix = matrix_identity()
        dx: float
        dz: float
        dx, dz = world_delta_to_hf_xz(Vector3(3.0, 9.0, -2.0), identity)
        self.assertAlmostEqual(dx, 3.0)
        self.assertAlmostEqual(dz, -2.0)


if __name__ == "__main__":
    unittest.main()
