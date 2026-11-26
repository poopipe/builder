"""road sections layer: classify, shp import, outliner, scene round-trip"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyray import Vector3
import shapefile

from builder.heightfield.heightfield_types import (
    RoadSectionsHeightSource,
    default_heightfield,
)
from builder.heightfield.road_sections.edit import (
    append_road_sections_edit_layer,
    import_road_shp,
)
from builder.heightfield.road_sections.overlay import (
    road_sections_overlay_cache_key,
    selected_usage_group_ids,
    usage_overlay_color,
)
from builder.heightfield.road_sections.road_sections_types import (
    classify_road_usage,
    road_usage_polyline_count,
)
from builder.heightfield.sample import sample_height
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.io.scene_format import (
    SceneCameraState,
    SceneDocument,
    SceneUiState,
    dumps_document,
    loads_document,
    node_from_json,
    node_to_json,
    scene_format_version,
)
from builder.scene.scene_types import Node, builtin_cube, transform_at
from builder.ui.outliner import group_children_map


def write_fixture_shp(directory: Path) -> Path:
    """tiny polyline shapefile with mixed highway classes"""
    path: Path = directory / "roads"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYLINE) as writer:
        writer.field("highway", "C", size=32)
        writer.line([[[-10.0, 0.0], [10.0, 0.0]]])
        writer.record("primary")
        writer.line([[[0.0, -10.0], [0.0, 10.0]]])
        writer.record("footway")
        writer.line([[[5.0, 5.0], [6.0, 6.0]]])
        writer.record("raceway")
    return path.with_suffix(".shp")


class TestClassifyRoadUsage(unittest.TestCase):
    def test_prefers_layer_then_highway(self) -> None:
        self.assertEqual(classify_road_usage("primary", ""), "primary")
        self.assertEqual(classify_road_usage("unfiltered", "trunk"), "trunk")
        self.assertEqual(classify_road_usage("", "residential"), "residential")
        self.assertIsNone(classify_road_usage("raceway", ""))
        self.assertIsNone(classify_road_usage("", "unknown_class"))

    def test_usage_colors_differ_for_road_vs_path(self) -> None:
        primary = usage_overlay_color("primary")
        path = usage_overlay_color("path")
        self.assertNotEqual(
            (primary.r, primary.g, primary.b),
            (path.r, path.g, path.b),
        )


class TestUsageOverlayFilter(unittest.TestCase):
    def test_usage_selection_isolates_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shp: Path = write_fixture_shp(Path(tmp))
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            _, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            layer: Node = layer_nodes[0]
            imported: list[Node] = import_road_shp(
                layer.id, shp, heightfield
            )
            nodes: dict[str, Node] = {layer.id: layer}
            for node in imported:
                nodes[node.id] = node
            usage = next(
                node
                for node in imported
                if node.road_usage_group is not None
            )
            shp_node = next(
                node for node in imported if node.road_shp is not None
            )
            self.assertEqual(
                selected_usage_group_ids(nodes, layer.id, [usage.id]),
                {usage.id},
            )
            self.assertIsNone(
                selected_usage_group_ids(nodes, layer.id, [shp_node.id])
            )
            self.assertIsNone(
                selected_usage_group_ids(nodes, layer.id, [layer.id])
            )

    def test_overlay_cache_key_includes_height_only_when_draping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shp: Path = write_fixture_shp(Path(tmp))
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            _, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            layer: Node = layer_nodes[0]
            imported: list[Node] = import_road_shp(
                layer.id, shp, heightfield
            )
            nodes: dict[str, Node] = {layer.id: layer}
            for node in imported:
                nodes[node.id] = node
            usage = next(
                node
                for node in imported
                if node.road_usage_group is not None
            )
            all_key = road_sections_overlay_cache_key(
                "hf1",
                layer.id,
                None,
                0.0,
                nodes,
            )
            usage_key = road_sections_overlay_cache_key(
                "hf1",
                layer.id,
                {usage.id},
                0.0,
                nodes,
            )
            self.assertIsNone(all_key[2])
            self.assertEqual(usage_key[2], (usage.id,))
            self.assertEqual(
                road_sections_overlay_cache_key(
                    "hf1",
                    layer.id,
                    {usage.id},
                    0.0,
                    nodes,
                ),
                usage_key,
            )


class TestRoadSectionsImport(unittest.TestCase):
    def test_import_groups_by_usage_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shp: Path = write_fixture_shp(Path(tmp))
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            updated, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            self.assertEqual(len(updated.edit_layers), 1)
            match updated.edit_layers[0].height:
                case RoadSectionsHeightSource(layer_node_id=layer_id):
                    pass
                case _:
                    self.fail("expected RoadSectionsHeightSource")
            layer: Node = layer_nodes[0]
            self.assertEqual(layer.id, layer_id)
            self.assertIsNotNone(layer.road_sections_layer)
            imported: list[Node] = import_road_shp(layer.id, shp, updated)
            shp_nodes: list[Node] = [
                node for node in imported if node.road_shp is not None
            ]
            self.assertEqual(len(shp_nodes), 1)
            usage_nodes: list[Node] = [
                node for node in imported if node.road_usage_group is not None
            ]
            usage_names: set[str] = {
                node.road_usage_group.usage
                for node in usage_nodes
                if node.road_usage_group is not None
            }
            self.assertEqual(usage_names, {"primary", "footway"})
            total: int = sum(
                road_usage_polyline_count(node.road_usage_group)
                for node in usage_nodes
                if node.road_usage_group is not None
            )
            self.assertEqual(total, 2)

    def test_import_remaps_map_space_to_hf_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shp: Path = write_fixture_shp(Path(tmp))
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            _, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            layer: Node = layer_nodes[0]
            imported: list[Node] = import_road_shp(
                layer.id, shp, heightfield
            )
            primary = next(
                node
                for node in imported
                if node.road_usage_group is not None
                and node.road_usage_group.usage == "primary"
            )
            assert primary.road_usage_group is not None
            points = primary.road_usage_group.polylines[0].points
            # east-west mirrored to match asc column order
            self.assertAlmostEqual(points[0][0], -(-10.0 - 128.0))
            self.assertAlmostEqual(points[0][1], 0.0 - 128.0)
            self.assertAlmostEqual(points[1][0], -(10.0 - 128.0))
            self.assertAlmostEqual(points[1][1], 0.0 - 128.0)

    def test_outliner_shows_usage_under_shp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shp: Path = write_fixture_shp(Path(tmp))
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            _, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            layer: Node = layer_nodes[0]
            imported: list[Node] = import_road_shp(
                layer.id, shp, heightfield
            )
            nodes: dict[str, Node] = {
                "hf1": Node(
                    id="hf1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=default_heightfield(),
                    name="Heightfield",
                ),
                layer.id: layer,
            }
            for node in imported:
                nodes[node.id] = node
            children = group_children_map(nodes)
            under_hf: list[Node] = children.get("hf1", [])
            self.assertEqual(len(under_hf), 1)
            self.assertIsNotNone(under_hf[0].road_sections_layer)
            under_layer: list[Node] = children.get(layer.id, [])
            self.assertEqual(len(under_layer), 1)
            self.assertIsNotNone(under_layer[0].road_shp)
            shp_id: str = under_layer[0].id
            under_shp: list[Node] = children.get(shp_id, [])
            self.assertEqual(len(under_shp), 2)
            self.assertTrue(
                all(node.road_usage_group is not None for node in under_shp)
            )

    def test_sample_height_ignores_road_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog: HeightmapCatalog = HeightmapCatalog()
            heightfield = default_heightfield()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            updated, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            before: float = sample_height(catalog, heightfield, 0.5, 0.5)
            after: float = sample_height(catalog, updated, 0.5, 0.5)
            self.assertAlmostEqual(before, after)
            self.assertEqual(len(layer_nodes), 1)

    def test_scene_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            updated, layer_nodes = append_road_sections_edit_layer(
                catalog, heightfield, "hf1", cache_dir=cache_dir
            )
            layer: Node = layer_nodes[0]
            shp: Path = write_fixture_shp(Path(tmp))
            imported: list[Node] = import_road_shp(layer.id, shp, updated)
            hf_node: Node = Node(
                id="hf1",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=updated,
                name="Heightfield",
            )
            all_nodes: list[Node] = [hf_node, layer, *imported]
            encoded = [node_to_json(node) for node in all_nodes]
            decoded = [node_from_json(payload) for payload in encoded]
            self.assertIsNotNone(decoded[1].road_sections_layer)
            shp_decoded = next(
                node for node in decoded if node.road_shp is not None
            )
            assert shp_decoded.road_shp is not None
            self.assertTrue(shp_decoded.road_shp.source_path.endswith(".shp"))
            usage_names: set[str] = {
                node.road_usage_group.usage
                for node in decoded
                if node.road_usage_group is not None
            }
            self.assertEqual(usage_names, {"primary", "footway"})
            self.assertEqual(scene_format_version, 39)
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
            text: str = dumps_document(document, {}, {}, {})
            loaded, _warnings = loads_document(text)
            self.assertEqual(loaded.version, 39)


if __name__ == "__main__":
    unittest.main()
