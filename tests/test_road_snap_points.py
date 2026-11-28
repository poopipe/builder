"""road-tool snap point extraction from odol road parts"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path

from tools.build_road_kit_widths import (
    _xz_dist,
    extract_road_snap_points,
    parse_snap_selection_indices,
    recover_snap_points_from_indices,
    snap_mapping_plausible,
)

_ASF1_25: Path = Path(r"P:\DZ\structures\roads\parts\asf1_25.p3d")
_ASF1_10_25: Path = Path(r"P:\DZ\structures\roads\parts\asf1_10 25.p3d")
_ASF3_22_50: Path = Path(r"P:\DZ\structures\roads\parts\asf3_22 50.p3d")
_ASF3_30_25: Path = Path(r"P:\DZ\structures\roads\parts\asf3_30 25.p3d")


def _pack_sel(name: str, vertex_index: int) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + struct.pack("<II", 0, 0)
        + b"\x00"
        + struct.pack("<II", 0, 1)
        + struct.pack("<H", vertex_index)
        + struct.pack("<I", 0)
    )


def _synthetic_road_bytes(
    points: list[tuple[float, float, float]],
    indices: dict[str, int],
) -> bytes:
    """minimal blob: count+selections, then a vertex table"""
    body: bytes = struct.pack("<I", 4)
    name: str
    for name in ("lb", "le", "pb", "pe"):
        body += _pack_sel(name, indices[name])
    for x, y, z in points:
        body += struct.pack("<fff", x, y, z)
    return body


class TestParseSnapSelections(unittest.TestCase):
    def test_parse_indices(self) -> None:
        data: bytes = _synthetic_road_bytes(
            [(0.0, 0.0, 0.0)] * 4,
            {"lb": 3, "le": 2, "pb": 0, "pe": 1},
        )
        parsed = parse_snap_selection_indices(data)
        self.assertEqual(parsed, {"lb": 3, "le": 2, "pb": 0, "pe": 1})


class TestRecoverSnapPoints(unittest.TestCase):
    def test_recover_straight_table(self) -> None:
        points: list[tuple[float, float, float]] = [
            (6.0, 0.0, -12.5),
            (6.0, 0.0, 12.5),
            (-6.0, 0.0, 12.5),
            (-6.0, 0.0, -12.5),
        ]
        indices: dict[str, int] = {"lb": 3, "le": 2, "pb": 0, "pe": 1}
        data: bytes = _synthetic_road_bytes(points, indices)
        snaps = recover_snap_points_from_indices(data, indices, width_m=12.0)
        self.assertIsNotNone(snaps)
        assert snaps is not None
        self.assertAlmostEqual(snaps["lb"][0], -6.0)
        self.assertAlmostEqual(snaps["lb"][2], -12.5)
        self.assertAlmostEqual(snaps["pb"][0], 6.0)
        self.assertAlmostEqual(snaps["pe"][2], 12.5)

    def test_reject_collapsed_lb_le_origin(self) -> None:
        """asf3_30 25-style false table: lb/le at origin, edge along travel"""
        mapped: dict[str, tuple[float, float, float]] = {
            "lb": (0.0, 0.0, 0.0),
            "le": (0.0, 0.0, 0.0),
            "pb": (1.440226674079895, 0.0, 7.120455265045166),
            "pe": (-1.440226674079895, 0.0, -7.120455265045166),
        }
        self.assertFalse(snap_mapping_plausible(mapped))
        points: list[tuple[float, float, float]] = [
            mapped["pb"],
            mapped["pe"],
            mapped["le"],
            mapped["lb"],
        ]
        indices: dict[str, int] = {"lb": 3, "le": 2, "pb": 0, "pe": 1}
        data: bytes = _synthetic_road_bytes(points, indices)
        self.assertIsNone(
            recover_snap_points_from_indices(data, indices, width_m=7.0)
        )

    def test_accept_corner_like_table(self) -> None:
        mapped: dict[str, tuple[float, float, float]] = {
            "lb": (-7.27, 0.0, -7.75),
            "le": (-3.12, 0.0, 7.75),
            "pb": (4.73, 0.0, -7.75),
            "pe": (7.27, 0.0, 1.75),
        }
        self.assertTrue(snap_mapping_plausible(mapped))


@unittest.skipUnless(_ASF1_25.is_file(), "dayz road parts not on P:")
class TestLiveRoadSnapPoints(unittest.TestCase):
    def test_asf1_25(self) -> None:
        snaps = extract_road_snap_points(_ASF1_25.read_bytes(), width_m=12.0)
        self.assertIsNotNone(snaps)
        assert snaps is not None
        self.assertAlmostEqual(snaps["lb"][0], -6.0, places=3)
        self.assertAlmostEqual(snaps["lb"][2], -12.5, places=3)
        self.assertAlmostEqual(snaps["pb"][0], 6.0, places=3)
        self.assertAlmostEqual(snaps["le"][2], 12.5, places=3)
        self.assertAlmostEqual(snaps["pe"][2], 12.5, places=3)

    def test_asf1_corner(self) -> None:
        if not _ASF1_10_25.is_file():
            self.skipTest("corner missing")
        snaps = extract_road_snap_points(
            _ASF1_10_25.read_bytes(), width_m=12.0
        )
        self.assertIsNotNone(snaps)
        assert snaps is not None
        # begin edge ~width 12 at negative z
        self.assertAlmostEqual(snaps["lb"][2], snaps["pb"][2], places=3)
        self.assertLess(snaps["lb"][2], 0.0)
        width_b: float = abs(snaps["pb"][0] - snaps["lb"][0])
        self.assertAlmostEqual(width_b, 12.0, delta=0.5)

    def test_asf3_corner_from_memory_vertex_table(self) -> None:
        if not _ASF3_22_50.is_file():
            self.skipTest("asf3_22 50 missing")
        snaps = extract_road_snap_points(
            _ASF3_22_50.read_bytes(), width_m=7.0
        )
        self.assertIsNotNone(snaps)
        assert snaps is not None
        self.assertAlmostEqual(
            _xz_dist(snaps["lb"], snaps["pb"]), 7.0, delta=0.05
        )
        self.assertAlmostEqual(
            _xz_dist(snaps["le"], snaps["pe"]), 7.0, delta=0.05
        )
        self.assertLess(snaps["lb"][2], 0.0)
        self.assertTrue(snap_mapping_plausible(snaps))

    def test_asf3_30_25_from_memory_vertex_table(self) -> None:
        if not _ASF3_30_25.is_file():
            self.skipTest("asf3_30 25 missing")
        snaps = extract_road_snap_points(
            _ASF3_30_25.read_bytes(), width_m=7.0
        )
        self.assertIsNotNone(snaps)
        assert snaps is not None
        self.assertAlmostEqual(
            _xz_dist(snaps["lb"], snaps["pb"]), 7.0, delta=0.05
        )
        self.assertTrue(snap_mapping_plausible(snaps))


if __name__ == "__main__":
    unittest.main()
