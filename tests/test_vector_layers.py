"""vector height layers: shapes, rasterize, hybrid link, scene round-trip"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pyray import Camera3D, CameraProjection, Ray, Vector3

from builder.generators.param_types import BoolParam, FloatParam
from builder.heightfield.heightfield_types import (
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
    ProceduralHeightSource,
    ProceduralHeightType,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
    default_heightfield,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.paint import create_paint_heightmap
from builder.heightfield.sample import sample_height
from builder.heightfield.vector.circle import CircleShapeParams, circle_rasterize
from builder.heightfield.vector.coords import (
    vector_layer_plane_y,
    vector_point_xz,
    vector_point_xzy,
)
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
    point_cube_size_for_line_radius,
)
from builder.heightfield.vector.edit import (
    append_polyline_point,
    append_vector_edit_layer,
    append_vector_road_edit_layer,
    append_vector_sculpt_edit_layer,
    delete_vector_points,
    strip_vector_sources_from_heightfield,
)
from builder.path.path_types import PathPointRecipe
from builder.path.points import default_path_point_recipe
from builder.heightfield.vector.pick import pick_nearest_vector_point
from builder.heightfield.vector.road import (
    absolute_road_points,
    blend_ribbon_height,
    drape_ribbon_height,
    nearest_segment_lerp_y,
    ribbon_height_polyline,
    road_half_width_points,
    road_offset_points,
)
from builder.heightfield.vector.stamp import (
    apply_weighted_sample,
    stamp_brush_at_uv,
)
from builder.scene.selection import subtree_ids
from builder.heightfield.vector.polyline import PolylineShapeParams, polyline_rasterize
from builder.heightfield.vector.rasterize import regenerate_vector_sources
from builder.heightfield.vector.rectangle import (
    RectangleShapeParams,
    rectangle_rasterize,
)
from builder.heightfield.vector.registry import get_vector_shape_spec, vector_shape_specs
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorLayerRecipe,
    VectorRasterContext,
)
from builder.view.gizmo import snap_vector_point_translation
from builder.io.scene_format import (
    dumps_document,
    loads_document,
    node_from_json,
    node_to_json,
    scene_format_version,
    SceneCameraState,
    SceneDocument,
    SceneUiState,
)
from builder.scene.scene_types import Node, NodeRole, builtin_cube, transform_at


class TestVectorRegistry(unittest.TestCase):
    def test_registered_kinds(self) -> None:
        kinds: set[str] = {spec.kind for spec in vector_shape_specs}
        self.assertEqual(kinds, {"polyline", "circle", "rectangle"})
        self.assertEqual(get_vector_shape_spec("polyline").label, "Polyline")


class TestVectorRasterize(unittest.TestCase):
    def test_polyline_stroke_raises_samples(self) -> None:
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=4.0,
            offset=1.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="add",
        )
        polyline_rasterize(
            context,
            [(-16.0, 0.0), (16.0, 0.0)],
            PolylineShapeParams(),
        )
        self.assertGreater(max(samples), 0.0)

    def test_circle_fill_raises_center(self) -> None:
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=2.0,
            offset=1.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.fill,
            brush_kind="add",
        )
        circle_rasterize(
            context,
            [(0.0, 0.0)],
            CircleShapeParams(radius=FloatParam(8.0, label="Radius")),
        )
        center: int = (height // 2) * width + (width // 2)
        self.assertGreater(samples[center], 0.5)

    def test_rectangle_fill_raises_interior(self) -> None:
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=1.0,
            offset=1.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.fill,
            brush_kind="add",
        )
        rectangle_rasterize(
            context,
            [(-10.0, -10.0), (10.0, 10.0)],
            RectangleShapeParams(),
        )
        center = (height // 2) * width + (width // 2)
        self.assertGreater(samples[center], 0.5)


class TestVectorHybridLink(unittest.TestCase):
    def test_append_creates_nodes_and_edit_layer(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield: Heightfield = default_heightfield()
            hf_id: str = "hf1"
            updated, nodes = append_vector_edit_layer(
                catalog,
                heightfield,
                hf_id,
                cache_dir=Path(tmp),
            )
            self.assertEqual(len(updated.edit_layers), 1)
            match updated.edit_layers[0].height:
                case VectorHeightSource(
                    layer_node_id=layer_node_id, heightmap_id=heightmap_id
                ):
                    layer_nodes = [
                        node for node in nodes if node.vector_layer is not None
                    ]
                    self.assertEqual(len(layer_nodes), 1)
                    self.assertEqual(layer_nodes[0].id, layer_node_id)
                    self.assertIn(heightmap_id, catalog.buffers)
                case _:
                    self.fail("expected VectorHeightSource")
            shape_nodes = [node for node in nodes if node.vector_shape is not None]
            self.assertEqual(len(shape_nodes), 1)
            points = [
                node for node in nodes if node.role is NodeRole.path_point
            ]
            self.assertGreaterEqual(len(points), 2)

    def test_move_point_changes_buffer(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            hf_id = "hf1"
            updated, new_nodes = append_vector_edit_layer(
                catalog,
                heightfield,
                hf_id,
                cache_dir=Path(tmp),
                default_shape_kind="circle",
            )
            nodes: dict[str, Node] = {
                hf_id: Node(
                    id=hf_id,
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                )
            }
            node: Node
            for node in new_nodes:
                nodes[node.id] = node
            regenerate_vector_sources(catalog, updated, nodes)
            source = updated.edit_layers[0].height
            assert isinstance(source, VectorHeightSource)
            before: float = max(catalog.buffers[source.heightmap_id].samples)
            # move circle center far away
            center = next(
                n for n in new_nodes if n.role is NodeRole.path_point
            )
            nodes[center.id] = replace(
                center,
                # default field is 256 m (origin-centered ±128)
                transform=transform_at(Vector3(200.0, 0.0, 200.0)),
            )
            regenerate_vector_sources(catalog, updated, nodes)
            after: float = max(catalog.buffers[source.heightmap_id].samples)
            # with center outside field, max contribution drops
            self.assertLess(after, before)

    def test_delete_vector_points_filters_role(self) -> None:
        nodes: dict[str, Node] = {
            "p": Node(
                id="p",
                parent_id="s",
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
            ),
            "g": Node(
                id="g",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
            ),
        }
        self.assertEqual(delete_vector_points(nodes, ["p", "g"]), ["p"])

    def test_append_polyline_point_extends_last_segment(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            _, new_nodes = append_vector_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
                default_shape_kind="polyline",
            )
            nodes: dict[str, Node] = {node.id: node for node in new_nodes}
            shape: Node = next(
                node for node in new_nodes if node.vector_shape is not None
            )
            before: list[Node] = [
                node
                for node in nodes.values()
                if node.parent_id == shape.id
                and node.role is NodeRole.path_point
            ]
            self.assertEqual(len(before), 3)
            last = before[-1].transform.translation
            prev = before[-2].transform.translation
            expected_x: float = float(last.x) + (
                float(last.x) - float(prev.x)
            )
            expected_z: float = float(last.z) + (
                float(last.z) - float(prev.z)
            )
            created: list[Node] = append_polyline_point(nodes, shape.id)
            added: Node = next(
                node for node in created if node.role is NodeRole.path_point
            )
            self.assertEqual(added.parent_id, shape.id)
            self.assertIs(added.role, NodeRole.path_point)
            self.assertEqual(added.name, "Point 3")
            self.assertAlmostEqual(
                float(added.transform.translation.x), expected_x
            )
            self.assertAlmostEqual(
                float(added.transform.translation.z), expected_z
            )
            self.assertGreaterEqual(len(created), 2)

    def test_remove_vector_layer_clears_nodes_and_source(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            hf_id = "hf1"
            updated, new_nodes = append_vector_edit_layer(
                catalog,
                heightfield,
                hf_id,
                cache_dir=Path(tmp),
            )
            nodes: dict[str, Node] = {
                hf_id: Node(
                    id=hf_id,
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                )
            }
            node: Node
            for node in new_nodes:
                nodes[node.id] = node
            source = updated.edit_layers[0].height
            assert isinstance(source, VectorHeightSource)
            layer_id: str = source.layer_node_id
            for remove_id in subtree_ids(nodes, [layer_id]):
                del nodes[remove_id]
            stripped: Heightfield = strip_vector_sources_from_heightfield(
                updated, {layer_id}
            )
            self.assertEqual(stripped.edit_layers, ())
            self.assertFalse(
                any(n.vector_layer is not None for n in nodes.values())
            )
            self.assertFalse(
                any(n.vector_shape is not None for n in nodes.values())
            )
            self.assertFalse(
                any(n.role is NodeRole.path_point for n in nodes.values())
            )


class TestVectorSceneRoundTrip(unittest.TestCase):
    def test_node_and_source_round_trip(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, new_nodes = append_vector_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            hf_node: Node = Node(
                id="hf1",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=updated,
                name="Heightfield",
            )
            all_nodes: list[Node] = [hf_node, *new_nodes]
            encoded = [node_to_json(node) for node in all_nodes]
            decoded = [node_from_json(payload) for payload in encoded]
            self.assertEqual(len(decoded), len(all_nodes))
            self.assertIsNotNone(decoded[0].heightfield)
            assert decoded[0].heightfield is not None
            match decoded[0].heightfield.edit_layers[0].height:
                case VectorHeightSource():
                    pass
                case _:
                    self.fail("expected vector source after round-trip")
            layer_decoded = next(
                node for node in decoded if node.vector_layer is not None
            )
            shape_decoded = next(
                node for node in decoded if node.vector_shape is not None
            )
            self.assertIsNotNone(layer_decoded.vector_layer)
            self.assertIsNotNone(shape_decoded.vector_shape)
            self.assertEqual(scene_format_version, 27)

    def test_document_version_is_27(self) -> None:
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
        self.assertEqual(loaded.version, 27)


class TestVectorSculpt(unittest.TestCase):
    def test_append_creates_sculpt_source(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, nodes = append_vector_sculpt_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            self.assertEqual(len(updated.edit_layers), 1)
            match updated.edit_layers[0].height:
                case VectorSculptHeightSource(
                    layer_node_id=layer_node_id, heightmap_id=heightmap_id
                ):
                    self.assertTrue(
                        any(
                            node.id == layer_node_id
                            and node.vector_layer is not None
                            for node in nodes
                        )
                    )
                    self.assertIn(heightmap_id, catalog.buffers)
                case _:
                    self.fail("expected VectorSculptHeightSource")

    def test_sculpt_flatten_raises_toward_target(self) -> None:
        width: int = 32
        height: int = 32
        samples: list[float] = [0.0] * (width * height)

        def below_at_uv(_u: float, _v: float) -> float:
            return 0.0

        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=8.0,
            offset=0.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="flatten",
            sculpt_metres=True,
            flatten_target=10.0,
            below_at_uv=below_at_uv,
        )
        stamp_brush_at_uv(context, 0.5, 0.5)
        center: int = (height // 2) * width + (width // 2)
        self.assertGreater(samples[center], 5.0)

    def test_sculpt_flatten_offset_raises_target(self) -> None:
        width: int = 32
        height: int = 32
        samples: list[float] = [0.0] * (width * height)

        def below_at_uv(_u: float, _v: float) -> float:
            return 0.0

        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=8.0,
            offset=5.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="flatten",
            sculpt_metres=True,
            flatten_target=0.0 + 5.0,
            below_at_uv=below_at_uv,
        )
        stamp_brush_at_uv(context, 0.5, 0.5)
        center: int = (height // 2) * width + (width // 2)
        self.assertGreater(samples[center], 2.0)

    def test_shape_offset_round_trip(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, new_nodes = append_vector_sculpt_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            shape: Node = next(
                node for node in new_nodes if node.vector_shape is not None
            )
            assert shape.vector_shape is not None
            shape = replace(
                shape,
                vector_shape=replace(
                    shape.vector_shape,
                    brush_kind="flatten",
                    offset=FloatParam(3.5, label="Offset", step=0.25),
                ),
            )
            encoded = node_to_json(shape)
            decoded = node_from_json(encoded)
            assert decoded.vector_shape is not None
            self.assertAlmostEqual(
                float(decoded.vector_shape.offset.value), 3.5
            )

    def test_negative_offset_lowers_samples(self) -> None:
        width: int = 32
        height: int = 32
        samples: list[float] = [0.0] * (width * height)
        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=8.0,
            offset=-2.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="add",
            sculpt_metres=True,
        )
        stamp_brush_at_uv(context, 0.5, 0.5)
        center: int = (height // 2) * width + (width // 2)
        self.assertLess(samples[center], -0.5)

    def test_sculpt_source_round_trip(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, new_nodes = append_vector_sculpt_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            hf_node: Node = Node(
                id="hf1",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=updated,
                name="Heightfield",
            )
            all_nodes: list[Node] = [hf_node, *new_nodes]
            encoded = [node_to_json(node) for node in all_nodes]
            decoded = [node_from_json(payload) for payload in encoded]
            assert decoded[0].heightfield is not None
            match decoded[0].heightfield.edit_layers[0].height:
                case VectorSculptHeightSource():
                    pass
                case _:
                    self.fail("expected vector sculpt source after round-trip")


class TestVectorWorkPlane(unittest.TestCase):
    def build_layer_scene(
        self, catalog: HeightmapCatalog, cache_dir: Path
    ) -> tuple[dict[str, Node], str, str, str]:
        """heightfield + vector layer + shape; returns nodes, hf, layer, shape ids"""
        heightfield = default_heightfield()
        updated, new_nodes = append_vector_edit_layer(
            catalog,
            heightfield,
            "hf1",
            cache_dir=cache_dir,
        )
        nodes: dict[str, Node] = {
            "hf1": Node(
                id="hf1",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=updated,
                name="Heightfield",
            )
        }
        node: Node
        for node in new_nodes:
            nodes[node.id] = node
        layer_id: str = next(
            node.id for node in new_nodes if node.vector_layer is not None
        )
        shape_id: str = next(
            node.id for node in new_nodes if node.vector_shape is not None
        )
        return nodes, "hf1", layer_id, shape_id

    def test_layer_y_sets_plane_height_without_moving_stamp_xz(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            nodes, hf_id, layer_id, shape_id = self.build_layer_scene(
                catalog, Path(tmp)
            )
            before: list[tuple[float, float]] = vector_point_xz(
                nodes, shape_id, hf_id
            )
            layer: Node = nodes[layer_id]
            nodes[layer_id] = replace(
                layer,
                transform=transform_at(Vector3(0.0, 12.5, 0.0)),
            )
            self.assertAlmostEqual(
                vector_layer_plane_y(nodes, layer_id, hf_id), 12.5
            )
            after: list[tuple[float, float]] = vector_point_xz(
                nodes, shape_id, hf_id
            )
            self.assertEqual(len(before), len(after))
            index: int
            for index in range(len(before)):
                self.assertAlmostEqual(before[index][0], after[index][0])
                self.assertAlmostEqual(before[index][1], after[index][1])

    def test_layer_xz_offsets_stamp_points(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            nodes, hf_id, layer_id, shape_id = self.build_layer_scene(
                catalog, Path(tmp)
            )
            before: list[tuple[float, float]] = vector_point_xz(
                nodes, shape_id, hf_id
            )
            layer: Node = nodes[layer_id]
            nodes[layer_id] = replace(
                layer,
                transform=transform_at(Vector3(4.0, 0.0, -3.0)),
            )
            after: list[tuple[float, float]] = vector_point_xz(
                nodes, shape_id, hf_id
            )
            self.assertAlmostEqual(after[0][0], before[0][0] + 4.0)
            self.assertAlmostEqual(after[0][1], before[0][1] - 3.0)

    def test_snap_vector_point_clears_local_y(self) -> None:
        point: Node = Node(
            id="p1",
            parent_id="shape",
            transform=transform_at(Vector3(2.0, 7.0, -1.0)),
            mesh_id=None,
            role=NodeRole.path_point,
        )
        snapped: Node = snap_vector_point_translation(point)
        self.assertAlmostEqual(float(snapped.transform.translation.x), 2.0)
        self.assertAlmostEqual(float(snapped.transform.translation.y), 0.0)
        self.assertAlmostEqual(float(snapped.transform.translation.z), -1.0)

    def test_pick_hits_point_cube_on_plane(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            nodes, hf_id, layer_id, shape_id = self.build_layer_scene(
                catalog, Path(tmp)
            )
            layer: Node = nodes[layer_id]
            nodes[layer_id] = replace(
                layer,
                transform=transform_at(Vector3(0.0, 5.0, 0.0)),
            )
            point: Node = next(
                node
                for node in nodes.values()
                if node.parent_id == shape_id
                and node.role is NodeRole.path_point
            )
            px: float = float(point.transform.translation.x)
            pz: float = float(point.transform.translation.z)
            ray: Ray = Ray(
                Vector3(px, 20.0, pz),
                Vector3(0.0, -1.0, 0.0),
            )
            camera: Camera3D = Camera3D()
            camera.position = Vector3(0.0, 100.0, 0.0)
            camera.target = Vector3(0.0, 0.0, 0.0)
            camera.up = Vector3(0.0, 0.0, 1.0)
            camera.fovy = 60.0
            camera.projection = CameraProjection.CAMERA_PERSPECTIVE
            viewport_height: float = 1440.0
            hit = pick_nearest_vector_point(
                nodes, ray, camera, viewport_height
            )
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit.point_id, point.id)
            line_radius: float = overlay_radius_for_camera(
                camera, 100.0, viewport_height, overlay_line_pixels
            )
            half: float = point_cube_size_for_line_radius(line_radius) * 0.5
            self.assertAlmostEqual(
                hit.distance, 20.0 - (5.0 + half), places=2
            )


class TestVectorRoad(unittest.TestCase):
    def test_append_creates_road_source(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, nodes = append_vector_road_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            self.assertEqual(len(updated.edit_layers), 1)
            match updated.edit_layers[0].height:
                case VectorRoadHeightSource(
                    layer_node_id=layer_node_id,
                    heightmap_id=heightmap_id,
                    coverage_id=coverage_id,
                ):
                    self.assertTrue(
                        any(
                            node.id == layer_node_id
                            and node.vector_layer is not None
                            for node in nodes
                        )
                    )
                    self.assertIn(heightmap_id, catalog.buffers)
                    self.assertIn(coverage_id, catalog.buffers)
                case _:
                    self.fail("expected VectorRoadHeightSource")

    def test_road_source_round_trip(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, new_nodes = append_vector_road_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            point: Node = next(
                node
                for node in new_nodes
                if node.role is NodeRole.path_point
            )
            point = replace(
                point,
                transform=transform_at(
                    Vector3(
                        float(point.transform.translation.x),
                        4.0,
                        float(point.transform.translation.z),
                    )
                ),
            )
            encoded = node_to_json(point)
            decoded = node_from_json(encoded)
            self.assertAlmostEqual(
                float(decoded.transform.translation.y), 4.0
            )
            self.assertIsNotNone(decoded.path_point)
            assert decoded.path_point is not None
            self.assertAlmostEqual(
                float(decoded.path_point.width_scale.value), 1.0
            )
            match updated.edit_layers[0].height:
                case VectorRoadHeightSource() as source:
                    from builder.io.scene_format import (
                        height_layer_source_from_json,
                        height_layer_source_to_json,
                    )

                    payload = height_layer_source_to_json(source)
                    restored = height_layer_source_from_json(
                        payload, "test"
                    )
                    self.assertIn("coverage_id", payload)
                    match restored:
                        case VectorRoadHeightSource(
                            layer_node_id=layer_node_id,
                            coverage_id=coverage_id,
                        ):
                            self.assertEqual(
                                layer_node_id, source.layer_node_id
                            )
                            self.assertEqual(
                                coverage_id, source.coverage_id
                            )
                        case _:
                            self.fail("expected VectorRoadHeightSource")
                case _:
                    self.fail("expected VectorRoadHeightSource")

    def test_absolute_road_points_add_surface_offset(self) -> None:
        def surface_at_xz(x: float, _z: float) -> float:
            return 10.0 if x < 0.0 else 20.0

        abs_points = absolute_road_points(
            [(-16.0, 0.0, 2.0), (16.0, 0.0, 3.0)],
            surface_at_xz,
        )
        self.assertAlmostEqual(abs_points[0][2], 12.0)
        self.assertAlmostEqual(abs_points[1][2], 23.0)

    def test_ribbon_flattens_to_interpolated_height(self) -> None:
        width: int = 32
        height: int = 32
        samples: list[float] = [0.0] * (width * height)
        coverage: list[float] = [0.0] * (width * height)

        def below_at_uv(_u: float, _v: float) -> float:
            return 0.0

        def ribbon_height(x: float, _z: float, _below: float) -> float:
            # absolute heights already (surface 0 + offsets 2 and 8)
            return nearest_segment_lerp_y(
                x,
                0.0,
                [(-16.0, 0.0, 2.0), (16.0, 0.0, 8.0)],
                closed=False,
            )

        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=4.0,
            offset=0.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="add",
            sculpt_metres=True,
            below_at_uv=below_at_uv,
            ribbon_height_at_xz=ribbon_height,
            coverage_samples=coverage,
        )
        # center texel on the path
        apply_weighted_sample(context, (height // 2) * width + (width // 2), 0.5, 0.5, 1.0)
        center: int = (height // 2) * width + (width // 2)
        self.assertAlmostEqual(samples[center], 5.0, places=2)
        self.assertAlmostEqual(coverage[center], 1.0, places=2)
        # far off the path remains zero when weight is 0
        apply_weighted_sample(context, 0, 0.0, 0.0, 0.0)
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(coverage[0], 0.0)

    def test_road_point_y_not_snapped(self) -> None:
        point: Node = Node(
            id="p1",
            parent_id="shape",
            transform=transform_at(Vector3(1.0, 3.0, 2.0)),
            mesh_id=None,
            role=NodeRole.path_point,
        )
        kept: Node = snap_vector_point_translation(point, allow_y=True)
        self.assertAlmostEqual(float(kept.transform.translation.y), 3.0)
        cleared: Node = snap_vector_point_translation(point, allow_y=False)
        self.assertAlmostEqual(float(cleared.transform.translation.y), 0.0)

    def test_point_offset_ignores_workplane_y(self) -> None:
        hf: Node = Node(
            id="hf1",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            heightfield=default_heightfield(),
        )
        layer: Node = Node(
            id="layer1",
            parent_id="hf1",
            transform=transform_at(Vector3(0.0, 50.0, 0.0)),
            mesh_id=None,
            vector_layer=VectorLayerRecipe(),
        )
        shape: Node = Node(
            id="shape1",
            parent_id="layer1",
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        point: Node = Node(
            id="p1",
            parent_id="shape1",
            transform=transform_at(Vector3(4.0, 2.0, 6.0)),
            mesh_id=None,
            role=NodeRole.path_point,
        )
        nodes: dict[str, Node] = {
            hf.id: hf,
            layer.id: layer,
            shape.id: shape,
            point.id: point,
        }
        points = vector_point_xzy(nodes, shape.id, hf.id)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][0], 4.0)
        self.assertAlmostEqual(points[0][1], 6.0)
        self.assertAlmostEqual(points[0][2], 2.0)
        abs_points = absolute_road_points(
            points, lambda _x, _z: 10.0
        )
        self.assertAlmostEqual(abs_points[0][2], 12.0)

    def test_open_polyline_fill_stamps_ribbon(self) -> None:
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=4.0,
            offset=1.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.fill,
            brush_kind="add",
        )
        polyline_rasterize(
            context,
            [(-16.0, 0.0), (16.0, 0.0)],
            PolylineShapeParams(closed=BoolParam(False, label="Closed")),
        )
        self.assertGreater(max(samples), 0.0)

    def test_road_ribbon_interior_is_flat(self) -> None:
        """center and mid-radius samples both reach full target height"""
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        coverage: list[float] = [0.0] * (width * height)
        target: float = 6.0

        def below_at_uv(_u: float, _v: float) -> float:
            return 0.0

        def ribbon_height(_x: float, _z: float, _below: float) -> float:
            return target

        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=8.0,
            offset=0.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="add",
            sculpt_metres=True,
            below_at_uv=below_at_uv,
            ribbon_height_at_xz=ribbon_height,
            coverage_samples=coverage,
        )
        polyline_rasterize(
            context,
            [(-24.0, 0.0), (24.0, 0.0)],
            PolylineShapeParams(closed=BoolParam(False, label="Closed")),
        )
        # texel at path center (0, 0)
        center_i: int = (height // 2) * width + (width // 2)
        # texel ~4m off the path (half of 8m half-width) along +z
        # z = (v - 0.5) * 64; want z ≈ 4 → v ≈ 0.5 + 4/64
        mid_v: float = 0.5 + 4.0 / 64.0
        mid_y: int = int(mid_v * height)
        mid_i: int = mid_y * width + (width // 2)
        self.assertAlmostEqual(samples[center_i], target, places=1)
        self.assertAlmostEqual(samples[mid_i], target, places=1)

    def test_vector_point_width_scale_round_trip(self) -> None:
        recipe: PathPointRecipe = replace(
            default_path_point_recipe(),
            width_scale=FloatParam(
                2.5, label="Width scale", step=0.05, minimum=0.05
            ),
        )
        point: Node = Node(
            id="p1",
            parent_id="shape",
            transform=transform_at(Vector3(1.0, 0.0, 2.0)),
            mesh_id=None,
            path_point=recipe,
            role=NodeRole.path_point,
        )
        decoded: Node = node_from_json(node_to_json(point))
        self.assertIsNotNone(decoded.path_point)
        assert decoded.path_point is not None
        self.assertAlmostEqual(
            float(decoded.path_point.width_scale.value), 2.5
        )

    def test_road_half_width_lerps_scales(self) -> None:
        points_xzw = road_half_width_points(
            [(-16.0, 0.0, 0.0, 1.0), (16.0, 0.0, 0.0, 2.0)],
            base_radius=4.0,
        )
        self.assertAlmostEqual(points_xzw[0][2], 4.0)
        self.assertAlmostEqual(points_xzw[1][2], 8.0)
        mid: float = nearest_segment_lerp_y(
            0.0, 0.0, points_xzw, closed=False
        )
        self.assertAlmostEqual(mid, 6.0)

    def test_road_stroke_respects_varying_width_scale(self) -> None:
        """samples near scale-1 end stay inside ~radius; far off scale-1 end stay 0"""
        width: int = 64
        height: int = 64
        samples: list[float] = [0.0] * (width * height)
        coverage: list[float] = [0.0] * (width * height)
        target: float = 5.0
        base_radius: float = 4.0
        points_xzw = road_half_width_points(
            [(-24.0, 0.0, 0.0, 1.0), (24.0, 0.0, 0.0, 2.0)],
            base_radius=base_radius,
        )

        def below_at_uv(_u: float, _v: float) -> float:
            return 0.0

        def ribbon_height(_x: float, _z: float, _below: float) -> float:
            return target

        def half_width_at(x: float, z: float) -> float:
            return nearest_segment_lerp_y(x, z, points_xzw, closed=False)

        context: VectorRasterContext = VectorRasterContext(
            samples=samples,
            width=width,
            height=height,
            size_x=64.0,
            size_z=64.0,
            radius_metres=max(hw for _x, _z, hw in points_xzw),
            offset=0.0,
            intensity=1.0,
            apply_mode=VectorApplyMode.stroke,
            brush_kind="add",
            sculpt_metres=True,
            below_at_uv=below_at_uv,
            ribbon_height_at_xz=ribbon_height,
            ribbon_half_width_at_xz=half_width_at,
            coverage_samples=coverage,
        )
        polyline_rasterize(
            context,
            [(-24.0, 0.0), (24.0, 0.0)],
            PolylineShapeParams(closed=BoolParam(False, label="Closed")),
        )
        # near left end (scale 1 → half-width 4): z≈2 should be stamped, z≈7 not
        left_u: float = 0.5 + (-20.0) / 64.0
        left_x: int = int(left_u * width)
        near_v: float = 0.5 + 2.0 / 64.0
        far_v: float = 0.5 + 7.0 / 64.0
        near_i: int = int(near_v * height) * width + left_x
        far_i: int = int(far_v * height) * width + left_x
        self.assertGreater(samples[near_i], 0.0)
        self.assertEqual(samples[far_i], 0.0)
        # near right end (scale 2 → half-width 8): z≈7 should still stamp
        right_u: float = 0.5 + 20.0 / 64.0
        right_x: int = int(right_u * width)
        wide_i: int = int(far_v * height) * width + right_x
        self.assertGreater(samples[wide_i], 0.0)

    def test_road_lerps_current_stack_below_to_absolute(self) -> None:
        """road coverage lerps accum toward stored absolute, not a baked delta"""
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir: Path = Path(tmp)
            height_id = create_paint_heightmap(
                catalog,
                label="road",
                width=4,
                height=4,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="road_cov",
                width=4,
                height=4,
                cache_dir=cache_dir,
            )
            catalog.buffers[height_id].samples[:] = [4.0] * 16
            catalog.buffers[coverage_id].samples[:] = [1.0] * 16
            bump: HeightEditLayer = HeightEditLayer(
                name="bump",
                height=ProceduralHeightSource(
                    ProceduralHeightType.constant, 10.0
                ),
                blend=HeightBlendMode.add,
                amplitude=1.0,
                offset=0.0,
            )
            road: HeightEditLayer = HeightEditLayer(
                name="road",
                height=VectorRoadHeightSource(
                    layer_node_id="layer",
                    heightmap_id=height_id,
                    coverage_id=coverage_id,
                ),
            )
            heightfield: Heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(0.0, label="Height Min"),
                base_height_max=FloatParam(0.0, label="Height Max"),
                edit_layers=(bump, road),
            )
            # lerp(10, 4, 1) = 4
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), 4.0
            )
            # raise the below stack without re-baking the road
            taller: HeightEditLayer = replace(
                bump,
                height=ProceduralHeightSource(
                    ProceduralHeightType.constant, 20.0
                ),
            )
            raised: Heightfield = replace(
                heightfield, edit_layers=(taller, road)
            )
            # still absolute 4 — not 4 + (20-10) from a stale delta
            self.assertAlmostEqual(
                sample_height(catalog, raised, 0.5, 0.5), 4.0
            )

    def test_road_zero_coverage_keeps_stack_below(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            height_id = create_paint_heightmap(
                catalog,
                label="road",
                width=4,
                height=4,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="road_cov",
                width=4,
                height=4,
                cache_dir=cache_dir,
            )
            catalog.buffers[height_id].samples[:] = [4.0] * 16
            catalog.buffers[coverage_id].samples[:] = [0.0] * 16
            bump = HeightEditLayer(
                name="bump",
                height=ProceduralHeightSource(
                    ProceduralHeightType.constant, 10.0
                ),
                blend=HeightBlendMode.add,
                amplitude=1.0,
                offset=0.0,
            )
            road = HeightEditLayer(
                name="road",
                height=VectorRoadHeightSource(
                    layer_node_id="layer",
                    heightmap_id=height_id,
                    coverage_id=coverage_id,
                ),
            )
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(0.0, label="Height Min"),
                base_height_max=FloatParam(0.0, label="Height Max"),
                edit_layers=(bump, road),
            )
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), 10.0
            )

    def test_grade_1_uses_knot_absolute_lerp(self) -> None:
        def surface_at_xz(x: float, _z: float) -> float:
            return 0.0 if x < 0.0 else 20.0

        knots = [(-16.0, 0.0, 0.0, 1.0), (16.0, 0.0, 0.0, 1.0)]
        offsets = road_offset_points(knots)
        abs_pts = absolute_road_points(
            [(x, z, y) for x, z, y, _s in knots], surface_at_xz
        )
        mid_x: float = 0.0
        mid_z: float = 0.0
        drape = drape_ribbon_height(
            mid_x, mid_z, "polyline", offsets, False, surface_at_xz
        )
        grade = ribbon_height_polyline(mid_x, mid_z, abs_pts, False)
        # mid surface under a step is 20 (x>=0); drape follows that
        self.assertAlmostEqual(drape, 20.0)
        # grade lerps knot absolutes 0 and 20 → 10
        self.assertAlmostEqual(grade, 10.0)
        self.assertAlmostEqual(blend_ribbon_height(drape, grade, 1.0), 10.0)
        self.assertAlmostEqual(blend_ribbon_height(drape, grade, 0.0), 20.0)
        self.assertAlmostEqual(blend_ribbon_height(drape, grade, 0.5), 15.0)

    def test_shape_grade_round_trip(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            updated, new_nodes = append_vector_road_edit_layer(
                catalog,
                heightfield,
                "hf1",
                cache_dir=Path(tmp),
            )
            shape: Node = next(
                node for node in new_nodes if node.vector_shape is not None
            )
            assert shape.vector_shape is not None
            shape = replace(
                shape,
                vector_shape=replace(
                    shape.vector_shape,
                    grade=replace(shape.vector_shape.grade, value=0.25),
                ),
            )
            decoded = node_from_json(node_to_json(shape))
            assert decoded.vector_shape is not None
            self.assertAlmostEqual(
                float(decoded.vector_shape.grade.value), 0.25
            )
            self.assertIsNotNone(updated.edit_layers)


if __name__ == "__main__":
    unittest.main()
