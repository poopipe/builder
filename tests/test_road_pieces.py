"""road piece catalog, fit, place, and clear"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.heightfield.road_sections.fit import (
    apply_yaw_xz,
    build_filleted_centerline,
    choose_straight_lengths,
    closest_on_polyline,
    corner_local_centerline_xz,
    corner_placement_yaw_deg,
    curve_merge_span_m,
    decompose_turn,
    fill_straights_between,
    fit_polyline_pieces,
    gap_should_merge_curves,
    max_fillet_radius_m,
    place_corner_stack_arc,
    stack_fillet_leg_m,
    vertex_distances,
    vertex_turns,
)
from builder.heightfield.road_sections.fit_report import (
    FitReportRow,
    flagged_rows,
    format_fit_report,
    report_polyline_placements,
    write_fit_report,
)
from builder.heightfield.road_sections.part_catalog import (
    CornerPiece,
    RoadKit,
    RoadPartsCatalog,
    build_kits,
    kit_for_usage,
    parse_stem,
    require_kit_for_usage,
)
from builder.heightfield.road_sections.kit_widths import PieceSnapPoints
from builder.heightfield.road_sections.proxies import (
    build_corner_proxy_geometry,
    corner_segment_count,
    road_proxy_mesh_id,
)
from builder.heightfield.road_sections.place import (
    drop_placements,
    place_polylines,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPiecePlacement,
    RoadPolyline,
    RoadUsageGroupRecipe,
)
from builder.io.scene_format import (
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    document_from_json,
    document_to_json,
    road_usage_group_from_json,
    road_usage_group_to_json,
    scene_format_version,
)
from builder.scene.scene_types import Node, builtin_cube, transform_at
from pyray import Vector3


def touch_p3d(parts: Path, stem: str) -> None:
    path: Path = parts / f"{stem}.p3d"
    path.write_bytes(b"")


class TestRoadPartCatalog(unittest.TestCase):
    def test_parse_straight_and_corner(self) -> None:
        straight = parse_stem("asf1_25")
        self.assertIsNotNone(straight)
        assert straight is not None
        self.assertEqual(straight.length_m, 25)
        corner = parse_stem("asf1_10 25")
        self.assertIsNotNone(corner)
        assert corner is not None
        self.assertEqual(corner.angle_deg, 10.0)
        self.assertEqual(corner.radius_m, 25.0)

    def test_build_kits_from_temp_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root: Path = Path(tmp) / "roads"
            parts: Path = root / "parts"
            parts.mkdir(parents=True)
            touch_p3d(parts, "asf1_25")
            touch_p3d(parts, "asf1_10")
            touch_p3d(parts, "asf1_6")
            touch_p3d(parts, "asf1_10 25")
            touch_p3d(parts, "asf1_6konec")
            kits = build_kits([root])
            self.assertIn("asf1", kits)
            kit: RoadKit = kits["asf1"]
            self.assertEqual(kit.straights_m, (25, 10, 6))
            self.assertEqual(kit.terminator, "asf1_6konec")
            self.assertEqual(len(kit.corners), 1)

    def test_kit_for_usage(self) -> None:
        self.assertEqual(kit_for_usage("primary"), "asf1")
        self.assertEqual(kit_for_usage("secondary"), "asf2")
        self.assertEqual(kit_for_usage("motorway"), "asf1")
        self.assertIsNone(kit_for_usage("raceway"))
        self.assertEqual(require_kit_for_usage("raceway"), "asf3")
        self.assertEqual(require_kit_for_usage("motorway"), "asf1")

    def test_proxy_mesh_id_includes_width(self) -> None:
        a = road_proxy_mesh_id("asf1_25", 12.0)
        b = road_proxy_mesh_id("asf1_25", 9.0)
        self.assertNotEqual(a, b)
        self.assertIn("w12.000", a.name)

    def test_width_for_kit_from_catalogue(self) -> None:
        catalog = RoadPartsCatalog()
        catalog.loaded = True
        catalog.kit_width_m = {"asf1": 12.0, "asf2": 9.0}
        self.assertAlmostEqual(catalog.width_for_kit("asf1"), 12.0)
        self.assertAlmostEqual(catalog.width_for_kit("asf2"), 9.0)
        self.assertAlmostEqual(catalog.width_for_kit("missing"), 6.0)


class TestRoadCornerProxy(unittest.TestCase):
    def test_corner_geometry_has_arc_rings(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        geo = build_corner_proxy_geometry(corner, 12.0)
        rings: int = corner_segment_count(90.0) + 1
        self.assertEqual(geo.vertex_count, rings * 4)
        self.assertEqual(geo.segments, corner_segment_count(90.0))
        self.assertGreater(geo.triangle_count, 0)
        # mid-arc origin: first centerline sample is not at world origin
        self.assertNotAlmostEqual(geo.positions[0], 0.0)
        self.assertNotAlmostEqual(geo.positions[2], 0.0)

    def test_corner_top_face_winds_for_plus_y(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        geo = build_corner_proxy_geometry(corner, 12.0)
        # first segment top tris are indices 6..11 (after bottom 0..5)
        i0: int = geo.indices[6]
        i1: int = geo.indices[7]
        i2: int = geo.indices[8]
        ax: float = geo.positions[i0 * 3]
        ay: float = geo.positions[i0 * 3 + 1]
        az: float = geo.positions[i0 * 3 + 2]
        bx: float = geo.positions[i1 * 3]
        by: float = geo.positions[i1 * 3 + 1]
        bz: float = geo.positions[i1 * 3 + 2]
        cx: float = geo.positions[i2 * 3]
        cy: float = geo.positions[i2 * 3 + 1]
        cz: float = geo.positions[i2 * 3 + 2]
        ny: float = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
        self.assertGreater(ny, 0.0)
        self.assertGreater(ay, 0.0)
        self.assertGreater(by, 0.0)
        self.assertGreater(cy, 0.0)

    def test_snap_proxy_pins_begin_and_end(self) -> None:
        corner = CornerPiece(angle_deg=10.0, radius_m=100.0, stem="asf1_10 100")
        snaps = PieceSnapPoints(
            lb=(-6.714035511016846, 0.0, -9.203353881835938),
            le=(-5.103657245635986, 0.0, 9.203353881835938),
            pb=(5.285964488983154, 0.0, -9.203353881835938),
            pe=(6.714035511016846, 0.0, 7.119575500488281),
        )
        geo = build_corner_proxy_geometry(corner, 12.0, snaps=snaps)
        begin: tuple[float, float] = snaps.begin_mid_xz()
        end: tuple[float, float] = snaps.end_mid_xz()
        # ring 0 left/right bottom → begin mid
        lx0: float = geo.positions[0]
        lz0: float = geo.positions[2]
        rx0: float = geo.positions[3]
        rz0: float = geo.positions[5]
        self.assertAlmostEqual(0.5 * (lx0 + rx0), begin[0], places=5)
        self.assertAlmostEqual(0.5 * (lz0 + rz0), begin[1], places=5)
        last: int = geo.segments * 4
        lx1: float = geo.positions[last * 3]
        lz1: float = geo.positions[last * 3 + 2]
        rx1: float = geo.positions[(last + 1) * 3]
        rz1: float = geo.positions[(last + 1) * 3 + 2]
        self.assertAlmostEqual(0.5 * (lx1 + rx1), end[0], places=5)
        self.assertAlmostEqual(0.5 * (lz1 + rz1), end[1], places=5)

    def test_snap_proxy_top_face_winds_for_plus_y(self) -> None:
        corner = CornerPiece(angle_deg=10.0, radius_m=100.0, stem="asf1_10 100")
        snaps = PieceSnapPoints(
            lb=(-6.714035511016846, 0.0, -9.203353881835938),
            le=(-5.103657245635986, 0.0, 9.203353881835938),
            pb=(5.285964488983154, 0.0, -9.203353881835938),
            pe=(6.714035511016846, 0.0, 7.119575500488281),
        )
        geo = build_corner_proxy_geometry(corner, 12.0, snaps=snaps)
        i0: int = geo.indices[6]
        i1: int = geo.indices[7]
        i2: int = geo.indices[8]
        ax: float = geo.positions[i0 * 3]
        ay: float = geo.positions[i0 * 3 + 1]
        az: float = geo.positions[i0 * 3 + 2]
        bx: float = geo.positions[i1 * 3]
        by: float = geo.positions[i1 * 3 + 1]
        bz: float = geo.positions[i1 * 3 + 2]
        cx: float = geo.positions[i2 * 3]
        cy: float = geo.positions[i2 * 3 + 1]
        cz: float = geo.positions[i2 * 3 + 2]
        ny: float = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
        self.assertGreater(ny, 0.0)
        self.assertGreater(ay, 0.0)


class TestRoadFit(unittest.TestCase):
    def test_choose_straight_slight_overshoot(self) -> None:
        # 14 m → prefer 10+6=16 (within 0.5*6 overshoot) over undershoot 6+6
        self.assertEqual(
            choose_straight_lengths(14.0, (25, 10, 6)), [10, 6]
        )
        self.assertEqual(
            sum(choose_straight_lengths(62.0, (25, 10, 6))), 62
        )

    def test_straight_fill_exact_cover(self) -> None:
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator="asf1_6konec",
            corners=(),
        )
        pts: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 62.0)]
        pieces = fit_polyline_pieces(pts, kit)
        stems: list[str] = [piece.stem for piece in pieces]
        self.assertEqual(stems.count("asf1_25"), 2)
        self.assertEqual(stems.count("asf1_6"), 2)
        self.assertEqual(stems[-1], "asf1_6konec")
        self.assertEqual(len(pieces), 5)

    def test_straight_pack_spans_full_segment(self) -> None:
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator=None,
            corners=(),
        )
        # 15 m → 10+6 with 1 m joint overlap; first starts at 0, last ends at 15
        pts: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 15.0)]
        pieces = fit_polyline_pieces(pts, kit)
        self.assertEqual(len(pieces), 2)
        self.assertEqual({p.stem for p in pieces}, {"asf1_10", "asf1_6"})
        # centers: 10 starts at 0 → center z=5; 6 ends at 15 → center z=12
        by_stem = {p.stem: p for p in pieces}
        self.assertAlmostEqual(by_stem["asf1_10"].z, 5.0, places=4)
        self.assertAlmostEqual(by_stem["asf1_6"].z, 12.0, places=4)

    def test_max_fillet_radius_from_legs(self) -> None:
        # 90° turn, 50 m legs → R_max = 49 / tan(45°) = 49
        r_max: float = max_fillet_radius_m(90.0, 50.0, 50.0)
        self.assertAlmostEqual(r_max, 49.0, places=5)

    def test_gentle_bend_uses_radius_not_angle_gate(self) -> None:
        # 5° bend used to be skipped by the 12° gate; radius fit should place
        corner = CornerPiece(angle_deg=5.0, radius_m=200.0, stem="asf1_5 200")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator=None,
            corners=(corner,),
        )
        # long legs so R=200 fits: leg = 200*tan(2.5°) ≈ 8.7 m
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, 100.0),
            (8.726535498, 199.6186),  # ~5° turn to the right
        ]
        turns = vertex_turns(pts)
        self.assertGreater(abs(turns[1]), 4.0)
        self.assertLess(abs(turns[1]), 6.0)
        stack = decompose_turn(turns[1], kit, 100.0, 100.0)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0][0].stem, "asf1_5 200")
        pieces = fit_polyline_pieces(pts, kit)
        self.assertTrue(any(p.stem == "asf1_5 200" for p in pieces))

    def test_straight_meets_arc_exit(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator=None,
            corners=(corner,),
        )
        # entry facing +Z; cw 90° exit should face +X
        placed, exit_x, exit_z, exit_yaw = place_corner_stack_arc(
            0.0, 0.0, 0.0, [(corner, 0.0)]
        )
        self.assertEqual(len(placed), 1)
        self.assertAlmostEqual(exit_yaw % 360.0, 90.0, places=4)
        # next straight starts exactly at the arc exit
        following, _pack_end = fill_straights_between(
            (exit_x, exit_z), (exit_x + 50.0, exit_z), kit
        )
        self.assertGreater(len(following), 0)
        first = following[0]
        # first straight center is 12.5 m along +X from exit for a 25 m piece
        self.assertAlmostEqual(first.x, exit_x + 12.5, places=4)
        self.assertAlmostEqual(first.z, exit_z, places=4)

    def test_ccw_corner_uses_yaw_spin_not_roll(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        # entry facing +Z; ccw 90° exit faces -X
        placed, exit_x, exit_z, exit_yaw = place_corner_stack_arc(
            0.0, 0.0, 0.0, [(corner, 180.0)]
        )
        self.assertEqual(len(placed), 1)
        piece = placed[0]
        self.assertAlmostEqual(piece.roll_deg, 180.0, places=5)
        self.assertAlmostEqual(
            piece.yaw_deg,
            corner_placement_yaw_deg(0.0, 90.0, clockwise=False),
            places=5,
        )
        self.assertAlmostEqual(piece.yaw_deg, 90.0, places=5)
        self.assertAlmostEqual(exit_x, -25.0, places=4)
        self.assertAlmostEqual(exit_z, 25.0, places=4)
        self.assertAlmostEqual(exit_yaw % 360.0, 270.0, places=4)
        # cw mesh under that yaw covers travel entry and exit
        entry_local = corner_local_centerline_xz(corner, 1.0)
        exit_local = corner_local_centerline_xz(corner, 0.0)
        entry_world = apply_yaw_xz(
            piece.x, piece.z, piece.yaw_deg, entry_local[0], entry_local[1]
        )
        exit_world = apply_yaw_xz(
            piece.x, piece.z, piece.yaw_deg, exit_local[0], exit_local[1]
        )
        self.assertAlmostEqual(entry_world[0], 0.0, places=4)
        self.assertAlmostEqual(entry_world[1], 0.0, places=4)
        self.assertAlmostEqual(exit_world[0], exit_x, places=4)
        self.assertAlmostEqual(exit_world[1], exit_z, places=4)

    def test_undercover_span_snaps_corner(self) -> None:
        # 25/12/6 cannot make 38–40; span 37.9 falls back to 37 → snap corner
        from math import tan, radians

        corner = CornerPiece(angle_deg=10.0, radius_m=100.0, stem="asf1_10 100")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 12, 6),
            terminator=None,
            corners=(corner,),
        )
        lengths = choose_straight_lengths(37.9, kit.straights_m)
        self.assertEqual(sum(lengths), 37)
        leg: float = 100.0 * tan(radians(5.0))
        approach: float = 37.9 + leg
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, approach),
            (17.633, approach + 99.0),  # ~10° turn
        ]
        pieces = fit_polyline_pieces(pts, kit)
        corner_i: int | None = next(
            (i for i, p in enumerate(pieces) if p.stem == "asf1_10 100"),
            None,
        )
        self.assertIsNotNone(corner_i)
        assert corner_i is not None
        self.assertGreater(corner_i, 0)
        prev = pieces[corner_i - 1]
        self.assertTrue(prev.stem.startswith("asf1_"))
        # corner should start near the under-covered pack end, not leave ~0.9 m
        filled, pack_end = fill_straights_between(
            (0.0, 0.0), (0.0, 37.9), kit
        )
        self.assertGreater(len(filled), 0)
        self.assertAlmostEqual(pack_end[1], 37.0, places=3)

    def test_short_gap_between_curves_merges(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=10.0, stem="asf1_90 10")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator=None,
            corners=(corner,),
        )
        # middle 15 m with 10 m legs → exit past next entry (~5 m < shortest 6)
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, 50.0),
            (15.0, 50.0),
            (15.0, 100.0),
        ]
        pieces = fit_polyline_pieces(pts, kit)
        corner_idxs: list[int] = [
            i for i, p in enumerate(pieces) if p.stem == "asf1_90 10"
        ]
        self.assertGreaterEqual(len(corner_idxs), 2)
        between: list[str] = [
            pieces[i].stem
            for i in range(corner_idxs[0] + 1, corner_idxs[1])
        ]
        self.assertEqual(between, [])

    def test_stub_between_fillets_merges(self) -> None:
        # ~6.9 m inter-fillet gap packs a lone 6 with gaps either side → merge
        corner = CornerPiece(angle_deg=90.0, radius_m=10.0, stem="asf1_90 10")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 12, 6),
            terminator=None,
            corners=(corner,),
        )
        self.assertTrue(gap_should_merge_curves(6.9, kit))
        # long under-cover must still place straights (do not jump the path)
        self.assertFalse(gap_should_merge_curves(37.9, kit))
        # middle long enough for legs but only a stub between exits/entries
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, 50.0),
            (18.0, 50.0),
            (18.0, 100.0),
        ]
        pieces = fit_polyline_pieces(pts, kit)
        corner_idxs: list[int] = [
            i for i, p in enumerate(pieces) if p.stem == "asf1_90 10"
        ]
        self.assertGreaterEqual(len(corner_idxs), 2)
        between: list[str] = [
            pieces[i].stem
            for i in range(corner_idxs[0] + 1, corner_idxs[1])
        ]
        self.assertEqual(between, [])

    def test_closest_on_polyline_snaps_and_tangent(self) -> None:
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
        ]
        x: float
        z: float
        yaw: float
        x, z, yaw = closest_on_polyline(pts, 2.0, 5.0)
        self.assertAlmostEqual(x, 0.0, places=4)
        self.assertAlmostEqual(z, 5.0, places=4)
        self.assertAlmostEqual(yaw % 360.0, 0.0, places=4)
        x, z, yaw = closest_on_polyline(pts, 5.0, 12.0)
        self.assertAlmostEqual(x, 5.0, places=4)
        self.assertAlmostEqual(z, 10.0, places=4)
        self.assertAlmostEqual(yaw % 360.0, 90.0, places=4)

    def test_fit_report_flags_short_gap(self) -> None:
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 10, 6),
            terminator=None,
            corners=(),
        )
        # two 10 m straights with centers 12 m apart → ~2 m junction gap
        placements = (
            RoadPiecePlacement("p0", "asf1_10", 0.0, 5.0, 0.0),
            RoadPiecePlacement("p0", "asf1_10", 0.0, 17.0, 0.0),
        )
        rows = report_polyline_placements("p0", placements, kit)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0].gap_ahead_m or 0.0, 2.0, places=4)
        self.assertIn("gap", rows[0].flags)
        self.assertIn("short", rows[0].flags)
        text = format_fit_report(
            rows, merge_span=curve_merge_span_m(kit)
        )
        self.assertIn("flagged: 1", text)
        self.assertEqual(len(flagged_rows(rows)), 1)

    def test_write_fit_report_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit_report.txt"
            path.write_text("stale\n", encoding="utf-8")
            rows = (
                FitReportRow(
                    polyline_id="p0",
                    index=0,
                    stem="asf1_10",
                    role="straight",
                    x=0.0,
                    z=0.0,
                    yaw_deg=0.0,
                    gap_ahead_m=None,
                    flags=(),
                ),
            )
            written = write_fit_report(rows, merge_span=5.0, path=path)
            self.assertEqual(written, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("road fit report", text)
            self.assertNotIn("stale", text)

    def test_filleted_centerline_inserts_arc_samples(self) -> None:
        corner = CornerPiece(angle_deg=90.0, radius_m=25.0, stem="asf1_90 25")
        kit = RoadKit(
            name="asf1",
            straights_m=(25, 6),
            terminator=None,
            corners=(corner,),
        )
        pts: list[tuple[float, float]] = [
            (0.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
        ]
        dists = vertex_distances(pts)
        turns = vertex_turns(pts)
        stack = decompose_turn(
            turns[1], kit, dists[1] - dists[0], dists[2] - dists[1]
        )
        self.assertTrue(stack)
        plans = {1: (stack, stack_fillet_leg_m(stack))}
        filleted = build_filleted_centerline(pts, plans, arc_samples=4)
        self.assertGreater(len(filleted), len(pts))


class TestRoadPlaceClear(unittest.TestCase):
    def _catalog(self) -> RoadPartsCatalog:
        with tempfile.TemporaryDirectory() as tmp:
            root: Path = Path(tmp) / "roads"
            parts: Path = root / "parts"
            parts.mkdir(parents=True)
            touch_p3d(parts, "asf1_25")
            touch_p3d(parts, "asf1_10")
            touch_p3d(parts, "asf1_6")
            touch_p3d(parts, "asf1_6konec")
            catalog = RoadPartsCatalog()
            catalog.ensure_loaded([root])
            return catalog

    def test_place_and_clear(self) -> None:
        catalog = self._catalog()
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                RoadPolyline(
                    id="a",
                    points=((0.0, 0.0), (0.0, 50.0)),
                ),
                RoadPolyline(
                    id="b",
                    points=((100.0, 0.0), (100.0, 50.0)),
                ),
            ),
        )
        placed = place_polylines(group, ["a"], catalog)
        a_count: int = sum(
            1 for p in placed.placements if p.polyline_id == "a"
        )
        b_count: int = sum(
            1 for p in placed.placements if p.polyline_id == "b"
        )
        self.assertGreater(a_count, 0)
        self.assertEqual(b_count, 0)
        cleared = drop_placements(placed, ["a"])
        self.assertEqual(cleared.placements, ())


class TestRoadPlacementRoundTrip(unittest.TestCase):
    def test_usage_group_placements_json(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                RoadPolyline(id="a", points=((0.0, 0.0), (10.0, 0.0))),
            ),
            placements=(
                RoadPiecePlacement(
                    polyline_id="a",
                    stem="asf1_10",
                    x=5.0,
                    z=0.0,
                    yaw_deg=90.0,
                    roll_deg=0.0,
                ),
            ),
            project_onto_shadow=True,
            ribbon_flatten=True,
        )
        payload = road_usage_group_to_json(group)
        self.assertTrue(payload["project_onto_shadow"])
        self.assertTrue(payload["ribbon_flatten"])
        decoded = road_usage_group_from_json(payload)
        self.assertEqual(decoded, group)

    def test_document_version_is_40(self) -> None:
        self.assertEqual(scene_format_version, 41)
        document = SceneDocument(
            version=scene_format_version,
            active_mesh_id=builtin_cube,
            ui=SceneUiState(
                side_panel_open=False,
                meshes_panel_open=False,
                height_layers_panel_open=False,
                sculpt_panel_open=False,
                outliner_open=False,
            ),
            camera=SceneCameraState(
                yaw=0.0,
                pitch=0.0,
                distance=10.0,
                target=Vector3(0.0, 0.0, 0.0),
                zoom_speed=1.0,
                move_speed=1.0,
                view_distance=1000.0,
            ),
            meshes=(),
            heightmaps=(),
            nodes=(
                Node(
                    id="u1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_usage_group=RoadUsageGroupRecipe(
                        usage="primary",
                        kit="asf1",
                        polylines=(
                            RoadPolyline(
                                id="a",
                                points=((0.0, 0.0), (10.0, 0.0)),
                            ),
                        ),
                        placements=(
                            RoadPiecePlacement(
                                polyline_id="a",
                                stem="asf1_10",
                                x=5.0,
                                z=0.0,
                                yaw_deg=90.0,
                            ),
                        ),
                    ),
                    name="primary",
                ),
            ),
        )
        payload = document_to_json(document, {}, {}, {})
        self.assertEqual(payload["version"], 40)
        decoded, warnings = document_from_json(payload)
        self.assertEqual(warnings, [])
        self.assertEqual(decoded.version, 40)
        node = decoded.nodes[0]
        assert node.road_usage_group is not None
        self.assertEqual(node.road_usage_group.kit, "asf1")
        self.assertEqual(len(node.road_usage_group.placements), 1)
        self.assertEqual(node.road_usage_group.placements[0].stem, "asf1_10")


if __name__ == "__main__":
    unittest.main()
