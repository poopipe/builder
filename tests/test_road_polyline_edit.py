"""road polyline ids, pick helpers, and edit ops"""

from __future__ import annotations

import unittest
from math import hypot

from builder.heightfield.road_sections.edit import (
    bridge_endpoints,
    delete_polylines,
    insert_vertex_between,
    merge_polylines,
    polyline_ids_shorter_than,
    polyline_length_m,
    set_polyline_vertex_xz,
    simplify_polylines,
    split_polyline_at_vertex,
    try_join_pair,
)
from builder.heightfield.road_sections.overlay import road_overlay_content_sig
from builder.heightfield.road_sections.place import drop_placements
from builder.heightfield.road_sections.pick import (
    clear_road_polyline_selection_if_stale,
    nearest_endpoint_ref,
    nearest_vertex_ref,
    polyline_distance_xz,
    segment_distance_xz,
    vertex_hit_is_better,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadEndpointRef,
    RoadPiecePlacement,
    RoadPolyline,
    RoadShpRecipe,
    RoadUsageGroupRecipe,
)
from builder.io.scene_format import (
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    document_from_json,
    document_to_json,
    road_polyline_from_json,
    road_polyline_to_json,
    scene_format_version,
)
from builder.scene.scene import Scene
from builder.scene.scene_types import Node, builtin_cube, transform_at
from builder.ui.road_sections_inspector import selected_usage_group_node
from builder.ui.ui_state import UiState
from pyray import Vector3


def usage_node(node_id: str, *polyline_ids: str) -> Node:
    polylines: tuple[RoadPolyline, ...] = tuple(
        line(polyline_id, (0.0, 0.0), (1.0, 0.0)) for polyline_id in polyline_ids
    )
    return Node(
        id=node_id,
        parent_id=None,
        transform=transform_at(Vector3(0.0, 0.0, 0.0)),
        mesh_id=None,
        road_usage_group=RoadUsageGroupRecipe(
            usage=node_id, kit="asf1", polylines=polylines
        ),
    )


def line(polyline_id: str, *points: tuple[float, float]) -> RoadPolyline:
    return RoadPolyline(id=polyline_id, points=points)


class TestRoadPolylineIds(unittest.TestCase):
    def test_scene_format_is_37(self) -> None:
        self.assertEqual(scene_format_version, 39)

    def test_polyline_json_round_trip(self) -> None:
        polyline: RoadPolyline = line("abc", (0.0, 0.0), (10.0, 0.0))
        payload = road_polyline_to_json(polyline)
        self.assertEqual(payload["id"], "abc")
        self.assertEqual(payload["points"], [[0.0, 0.0], [10.0, 0.0]])
        decoded = road_polyline_from_json(payload)
        self.assertEqual(decoded, polyline)

    def test_rejects_list_polyline(self) -> None:
        with self.assertRaises(ValueError):
            road_polyline_from_json([[0.0, 0.0], [1.0, 1.0]])


class TestRoadPolylineEdit(unittest.TestCase):
    def test_delete_polylines(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (1.0, 0.0)),
                line("b", (2.0, 0.0), (3.0, 0.0)),
            ),
        )
        updated = delete_polylines(group, ["a"])
        self.assertEqual([polyline.id for polyline in updated.polylines], ["b"])

    def test_simplify_keeps_id(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line(
                    "a",
                    (0.0, 0.0),
                    (5.0, 0.1),
                    (10.0, 0.0),
                ),
            ),
        )
        updated = simplify_polylines(group, ["a"], eps_m=1.0)
        self.assertEqual(updated.polylines[0].id, "a")
        self.assertEqual(len(updated.polylines[0].points), 2)

    def test_merge_joinable(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (10.0, 0.0)),
                line("b", (10.0, 0.0), (20.0, 0.0)),
            ),
        )
        updated, result_ids = merge_polylines(group, ["a", "b"], snap_m=0.1)
        self.assertEqual(len(updated.polylines), 1)
        self.assertEqual(len(result_ids), 1)
        self.assertEqual(
            updated.polylines[0].points,
            ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
        )

    def test_merge_leaves_gap_unjoined(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (10.0, 0.0)),
                line("b", (30.0, 0.0), (40.0, 0.0)),
            ),
        )
        updated, result_ids = merge_polylines(group, ["a", "b"], snap_m=0.1)
        self.assertEqual(len(updated.polylines), 2)
        self.assertEqual(set(result_ids), {"a", "b"})

    def test_merge_partial_within_selection(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (10.0, 0.0)),
                line("b", (10.0, 0.0), (20.0, 0.0)),
                line("c", (50.0, 0.0), (60.0, 0.0)),
                line("keep", (100.0, 0.0), (110.0, 0.0)),
            ),
        )
        updated, result_ids = merge_polylines(
            group, ["a", "b", "c"], snap_m=0.1
        )
        self.assertEqual(len(updated.polylines), 3)
        keep = next(p for p in updated.polylines if p.id == "keep")
        self.assertEqual(keep.points, ((100.0, 0.0), (110.0, 0.0)))
        self.assertEqual(len(result_ids), 2)
        self.assertIn("c", result_ids)
        merged = next(p for p in updated.polylines if p.id not in {"c", "keep"})
        self.assertEqual(
            merged.points, ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
        )

    def test_try_join_reverses_as_needed(self) -> None:
        a = line("a", (0.0, 0.0), (10.0, 0.0))
        b = line("b", (20.0, 0.0), (10.0, 0.0))
        joined = try_join_pair(a, b, snap_m=0.1)
        assert joined is not None
        self.assertEqual(
            joined.points, ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
        )

    def test_bridge_endpoints(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (9.0, 0.0)),
                line("b", (10.0, 0.0), (20.0, 0.0)),
            ),
        )
        updated = bridge_endpoints(
            group,
            RoadEndpointRef(polyline_id="a", vertex_index=1),
            RoadEndpointRef(polyline_id="b", vertex_index=0),
        )
        self.assertEqual(len(updated.polylines), 3)
        self.assertEqual(updated.polylines[0].points[-1], (9.0, 0.0))
        self.assertEqual(
            updated.polylines[-1].points, ((9.0, 0.0), (10.0, 0.0))
        )

    def test_polyline_length_m(self) -> None:
        self.assertAlmostEqual(
            polyline_length_m(line("a", (0.0, 0.0), (3.0, 4.0))),
            5.0,
        )

    def test_polyline_ids_shorter_than(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("short", (0.0, 0.0), (1.0, 0.0)),
                line("long", (0.0, 0.0), (10.0, 0.0)),
            ),
        )
        self.assertEqual(polyline_ids_shorter_than(group, 2.0), ["short"])
        self.assertEqual(polyline_ids_shorter_than(group, 0.5), [])

    def test_segment_distance(self) -> None:
        self.assertAlmostEqual(
            segment_distance_xz(5.0, 1.0, 0.0, 0.0, 10.0, 0.0), 1.0
        )
        self.assertAlmostEqual(
            polyline_distance_xz(
                5.0, 1.0, line("a", (0.0, 0.0), (10.0, 0.0))
            ),
            1.0,
        )

    def test_set_polyline_vertex_xz_keeps_placements(self) -> None:
        placement = RoadPiecePlacement(
            polyline_id="a",
            stem="straight",
            x=5.0,
            z=0.0,
            yaw_deg=0.0,
        )
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
            ),
            placements=(placement,),
        )
        updated = set_polyline_vertex_xz(group, "a", 1, 5.0, 2.0)
        self.assertEqual(
            updated.polylines[0].points,
            ((0.0, 0.0), (5.0, 2.0), (10.0, 0.0)),
        )
        self.assertEqual(updated.placements, (placement,))
        cleared = drop_placements(updated, ["a"])
        self.assertEqual(cleared.placements, ())

    def test_nearest_vertex_includes_midpoints(self) -> None:
        polyline = line("a", (0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
        mid = nearest_vertex_ref(5.0, 0.1, polyline, 1.0)
        self.assertIsNotNone(mid)
        assert mid is not None
        self.assertEqual(mid.vertex_index, 1)
        end_only = nearest_endpoint_ref(5.0, 0.1, polyline, 1.0)
        self.assertIsNone(end_only)

    def test_nearest_vertex_prefers_closer_of_two(self) -> None:
        polyline = line("a", (0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
        hit = nearest_vertex_ref(4.6, 0.0, polyline, 1.0)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.vertex_index, 1)

    def test_shared_vertex_tie_prefers_closer_body(self) -> None:
        # click slightly toward the right half after a split at (5,0)
        left = line("left", (0.0, 0.0), (5.0, 0.0))
        right = line("right", (5.0, 0.0), (10.0, 0.0))
        px: float = 5.2
        pz: float = 0.0
        left_vert: float = hypot(px - 5.0, pz - 0.0)
        right_vert: float = hypot(px - 5.0, pz - 0.0)
        self.assertAlmostEqual(left_vert, right_vert)
        self.assertTrue(
            vertex_hit_is_better(
                right_vert,
                polyline_distance_xz(px, pz, right),
                left_vert,
                polyline_distance_xz(px, pz, left),
            )
        )
        self.assertFalse(
            vertex_hit_is_better(
                left_vert,
                polyline_distance_xz(px, pz, left),
                right_vert,
                polyline_distance_xz(px, pz, right),
            )
        )

    def test_split_polyline_at_interior(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
                line("keep", (0.0, 1.0), (1.0, 1.0)),
            ),
            placements=(
                RoadPiecePlacement(
                    polyline_id="a",
                    stem="straight",
                    x=5.0,
                    z=0.0,
                    yaw_deg=0.0,
                ),
            ),
        )
        updated = split_polyline_at_vertex(group, "a", 1)
        self.assertEqual(len(updated.polylines), 3)
        self.assertEqual(updated.polylines[0].id, "a")
        self.assertEqual(
            updated.polylines[0].points, ((0.0, 0.0), (5.0, 0.0))
        )
        self.assertEqual(
            updated.polylines[1].points, ((5.0, 0.0), (10.0, 0.0))
        )
        self.assertEqual(updated.polylines[2].id, "keep")
        self.assertEqual(updated.placements, ())

    def test_split_rejects_end_vertex(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(line("a", (0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),),
        )
        with self.assertRaises(ValueError):
            split_polyline_at_vertex(group, "a", 0)
        with self.assertRaises(ValueError):
            split_polyline_at_vertex(group, "a", 2)

    def test_insert_vertex_between_adjacent(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(line("a", (0.0, 0.0), (10.0, 0.0)),),
            placements=(
                RoadPiecePlacement(
                    polyline_id="a",
                    stem="straight",
                    x=5.0,
                    z=0.0,
                    yaw_deg=0.0,
                ),
            ),
        )
        updated, new_ref = insert_vertex_between(
            group,
            RoadEndpointRef(polyline_id="a", vertex_index=0),
            RoadEndpointRef(polyline_id="a", vertex_index=1),
        )
        self.assertEqual(
            updated.polylines[0].points,
            ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        )
        self.assertEqual(new_ref.vertex_index, 1)
        self.assertEqual(updated.placements, ())

    def test_insert_vertex_between_non_adjacent(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (4.0, 0.0), (10.0, 0.0)),
            ),
        )
        updated, new_ref = insert_vertex_between(
            group,
            RoadEndpointRef(polyline_id="a", vertex_index=0),
            RoadEndpointRef(polyline_id="a", vertex_index=2),
        )
        self.assertEqual(
            updated.polylines[0].points,
            ((0.0, 0.0), (5.0, 0.0), (4.0, 0.0), (10.0, 0.0)),
        )
        self.assertEqual(new_ref.vertex_index, 1)

    def test_insert_rejects_different_polylines(self) -> None:
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (1.0, 0.0)),
                line("b", (2.0, 0.0), (3.0, 0.0)),
            ),
        )
        with self.assertRaises(ValueError):
            insert_vertex_between(
                group,
                RoadEndpointRef(polyline_id="a", vertex_index=0),
                RoadEndpointRef(polyline_id="b", vertex_index=0),
            )

    def test_overlay_content_sig_stable_when_live_polyline_omitted(self) -> None:
        layer = Node(
            id="layer",
            parent_id="hf",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        shp = Node(
            id="shp",
            parent_id="layer",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_shp=RoadShpRecipe(source_path="roads.shp"),
        )
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(
                line("a", (0.0, 0.0), (10.0, 0.0)),
                line("b", (0.0, 5.0), (10.0, 5.0)),
            ),
        )
        usage = Node(
            id="usage",
            parent_id="shp",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_usage_group=group,
        )
        nodes: dict[str, Node] = {
            layer.id: layer,
            shp.id: shp,
            usage.id: usage,
        }
        omit: frozenset[str] = frozenset({"a"})
        before = road_overlay_content_sig(
            nodes, layer.id, None, omit_polyline_ids=omit
        )
        moved = set_polyline_vertex_xz(group, "a", 1, 10.0, 3.0)
        nodes[usage.id] = Node(
            id="usage",
            parent_id="shp",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_usage_group=moved,
        )
        after = road_overlay_content_sig(
            nodes, layer.id, None, omit_polyline_ids=omit
        )
        self.assertEqual(before, after)

    def test_overlay_content_sig_changes_on_vertex_move(self) -> None:
        layer = Node(
            id="layer",
            parent_id="hf",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        shp = Node(
            id="shp",
            parent_id="layer",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_shp=RoadShpRecipe(source_path="roads.shp"),
        )
        group = RoadUsageGroupRecipe(
            usage="primary",
            kit="asf1",
            polylines=(line("a", (0.0, 0.0), (10.0, 0.0)),),
        )
        usage = Node(
            id="usage",
            parent_id="shp",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_usage_group=group,
        )
        nodes: dict[str, Node] = {
            layer.id: layer,
            shp.id: shp,
            usage.id: usage,
        }
        before = road_overlay_content_sig(nodes, layer.id, None)
        moved = set_polyline_vertex_xz(group, "a", 1, 10.0, 3.0)
        nodes[usage.id] = Node(
            id="usage",
            parent_id="shp",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            road_usage_group=moved,
        )
        after = road_overlay_content_sig(nodes, layer.id, None)
        self.assertNotEqual(before, after)


class TestRoadSelectionAlign(unittest.TestCase):
    def test_stale_when_sole_selection_differs(self) -> None:
        nodes: dict[str, Node] = {
            "a": usage_node("a", "p1"),
            "b": usage_node("b", "p2"),
        }
        self.assertTrue(
            clear_road_polyline_selection_if_stale(
                nodes, "a", ["p1"], ["b"]
            )
        )

    def test_not_stale_when_sole_matches(self) -> None:
        nodes: dict[str, Node] = {"a": usage_node("a", "p1")}
        self.assertFalse(
            clear_road_polyline_selection_if_stale(
                nodes, "a", ["p1"], ["a"]
            )
        )

    def test_stale_when_multi_selected(self) -> None:
        nodes: dict[str, Node] = {
            "a": usage_node("a", "p1"),
            "b": usage_node("b", "p2"),
        }
        self.assertTrue(
            clear_road_polyline_selection_if_stale(
                nodes, "a", ["p1"], ["a", "b"]
            )
        )

    def test_selected_usage_follows_sole_scene_selection(self) -> None:
        scene: Scene = Scene(
            nodes={
                "a": usage_node("a", "p1"),
                "b": usage_node("b", "p2"),
            },
            selected_ids=["b"],
        )
        selected: Node | None = selected_usage_group_node(scene)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.id, "b")

    def test_selected_usage_none_when_multi(self) -> None:
        scene: Scene = Scene(
            nodes={
                "a": usage_node("a", "p1"),
                "b": usage_node("b", "p2"),
            },
            selected_ids=["a", "b"],
        )
        self.assertIsNone(selected_usage_group_node(scene))

    def test_two_vertices_on_same_polyline(self) -> None:
        ui = UiState()
        ui.set_road_polyline_selection(
            "usage",
            "a",
            additive=False,
            endpoint=RoadEndpointRef(polyline_id="a", vertex_index=0),
        )
        ui.set_road_polyline_selection(
            "usage",
            "a",
            additive=True,
            endpoint=RoadEndpointRef(polyline_id="a", vertex_index=2),
        )
        self.assertEqual(len(ui.road_endpoints), 2)
        self.assertEqual(ui.road_endpoints[0].vertex_index, 0)
        self.assertEqual(ui.road_endpoints[1].vertex_index, 2)

    def test_connect_ends_needs_different_polylines(self) -> None:
        ui = UiState()
        ui.set_road_polyline_selection(
            "usage",
            "a",
            additive=False,
            endpoint=RoadEndpointRef(polyline_id="a", vertex_index=0),
        )
        ui.set_road_polyline_selection(
            "usage",
            "b",
            additive=True,
            endpoint=RoadEndpointRef(polyline_id="b", vertex_index=0),
        )
        self.assertEqual(len(ui.road_endpoints), 2)
        self.assertNotEqual(
            ui.road_endpoints[0].polyline_id, ui.road_endpoints[1].polyline_id
        )


class TestRoadPolylineDocument(unittest.TestCase):
    def test_usage_group_round_trip(self) -> None:
        document: SceneDocument = SceneDocument(
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
            nodes=(),
        )
        payload = document_to_json(document, {}, {}, {})
        self.assertEqual(payload["version"], 39)
        decoded, warnings = document_from_json(payload)
        self.assertEqual(warnings, [])
        self.assertEqual(decoded.version, 39)


if __name__ == "__main__":
    unittest.main()
