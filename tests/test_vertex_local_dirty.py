"""vertex-local dirty aabb and deferred gpu paint bumps"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyray import Vector3

from builder.generators.param_types import BoolParam, FloatParam
from builder.heightfield.heightfield_types import default_heightfield
from builder.heightfield.heightmap_catalog import (
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
)
from builder.heightfield.paint import (
    flush_deferred_paint_touches,
    tick_deferred_gpu_flush,
    touch_heightmap_buffer,
)
from builder.heightfield.vector.edit import append_vector_sculpt_edit_layer
from builder.heightfield.vector.circle import CircleShapeParams
from builder.heightfield.vector.polyline import PolylineShapeParams
from builder.heightfield.vector.rasterize import (
    dirty_aabb_for_vector_nodes,
    expand_dirty_for_layer,
    local_edit_shape_ids,
    point_neighborhood_aabb,
    shape_world_aabb,
)
from builder.heightfield.vector.stamp import (
    distance_to_segment_list,
    distance_to_segments,
    polyline_segments,
    segments_intersecting_aabb,
)
from builder.heightfield.vector.vector_types import (
    VectorApplyMode,
    VectorShapeRecipe,
)
from builder.path.points import (
    default_path_point_recipe,
    path_segments_near_point_ids,
)
from builder.scene.scene_types import Node, NodeRole, transform_at


class TestVertexLocalDirty(unittest.TestCase):
    def test_point_neighborhood_smaller_than_full_polyline(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            hf_id = "hf1"
            updated, new_nodes = append_vector_sculpt_edit_layer(
                catalog,
                heightfield,
                hf_id,
                cache_dir=Path(tmp),
                default_shape_kind="polyline",
            )
            nodes: dict[str, Node] = {
                hf_id: Node(
                    id=hf_id,
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                )
            }
            for node in new_nodes:
                nodes[node.id] = node
            shape = next(n for n in new_nodes if n.vector_shape is not None)
            assert shape.vector_shape is not None
            layer_id = shape.parent_id
            assert layer_id is not None
            nodes[shape.id] = Node(
                id=shape.id,
                parent_id=layer_id,
                transform=shape.transform,
                mesh_id=None,
                vector_shape=VectorShapeRecipe(
                    kind="polyline",
                    apply_mode=VectorApplyMode.stroke,
                    brush_kind=shape.vector_shape.brush_kind,
                    offset=shape.vector_shape.offset,
                    intensity=shape.vector_shape.intensity,
                    radius=FloatParam(4.0, label="Radius"),
                    grade=shape.vector_shape.grade,
                    params=PolylineShapeParams(
                        closed=BoolParam(False, label="Closed")
                    ),
                    enabled=True,
                ),
                name=shape.name,
            )
            # drop any default path points, then add a long open path
            for node_id in list(nodes.keys()):
                child = nodes[node_id]
                if (
                    child.parent_id == shape.id
                    and child.role is NodeRole.path_point
                ):
                    del nodes[node_id]
            p0_id = "pt0"
            p1_id = "pt1"
            p2_id = "pt2"
            nodes[p0_id] = Node(
                id=p0_id,
                parent_id=shape.id,
                transform=transform_at(Vector3(-80.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            )
            nodes[p1_id] = Node(
                id=p1_id,
                parent_id=shape.id,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            )
            nodes[p2_id] = Node(
                id=p2_id,
                parent_id=shape.id,
                transform=transform_at(Vector3(80.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            )
            # endpoint neighborhood is one segment (~80m + pad), not the full 160m path
            local = dirty_aabb_for_vector_nodes(nodes, [p0_id])
            self.assertIsNotNone(local)
            assert local is not None
            self.assertEqual(
                local, point_neighborhood_aabb(nodes, p0_id, hf_id)
            )
            self.assertLess(local[1] - local[0], 100.0)
            self.assertEqual(local_edit_shape_ids(nodes, [p0_id]), {shape.id})
            expanded, restamp = expand_dirty_for_layer(
                nodes,
                layer_id,
                hf_id,
                local,
                local_shape_ids={shape.id},
            )
            self.assertEqual(expanded, local)
            self.assertIn(shape.id, restamp)
            # whole-shape dirty still covers the far end
            full = shape_world_aabb(nodes, nodes[shape.id], hf_id)
            self.assertIsNotNone(full)
            assert full is not None
            self.assertGreater(full[1] - full[0], local[1] - local[0])


class TestExpandDirtyNoGrow(unittest.TestCase):
    def test_overlapping_sibling_restamped_without_growing_dirty(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            hf_id = "hf1"
            updated, new_nodes = append_vector_sculpt_edit_layer(
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
                )
            }
            for node in new_nodes:
                nodes[node.id] = node
            layer = next(n for n in new_nodes if n.vector_layer is not None)
            shape_a = next(n for n in new_nodes if n.vector_shape is not None)
            assert shape_a.vector_shape is not None
            # sibling whose stamp aabb overlaps a small seed near the origin
            shape_b_id = "shape_b"
            nodes[shape_b_id] = Node(
                id=shape_b_id,
                parent_id=layer.id,
                transform=transform_at(Vector3(8.0, 0.0, 0.0)),
                mesh_id=None,
                vector_shape=VectorShapeRecipe(
                    kind="circle",
                    apply_mode=VectorApplyMode.fill,
                    brush_kind=shape_a.vector_shape.brush_kind,
                    offset=shape_a.vector_shape.offset,
                    intensity=shape_a.vector_shape.intensity,
                    radius=FloatParam(4.0, label="Radius"),
                    grade=shape_a.vector_shape.grade,
                    params=CircleShapeParams(
                        radius=FloatParam(8.0, label="Radius")
                    ),
                    enabled=True,
                ),
                name="Circle B",
            )
            nodes["point_b"] = Node(
                id="point_b",
                parent_id=shape_b_id,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            )
            seed = (-5.0, 5.0, -5.0, 5.0)
            expanded, restamp = expand_dirty_for_layer(
                nodes,
                layer.id,
                hf_id,
                seed,
                local_shape_ids={shape_a.id},
            )
            self.assertEqual(expanded, seed)
            self.assertIn(shape_a.id, restamp)
            self.assertIn(shape_b_id, restamp)


class TestSegmentFilter(unittest.TestCase):
    def test_filtered_segments_match_full_distance_in_local_aabb(self) -> None:
        points: list[tuple[float, float]] = [
            (-100.0, 0.0),
            (-50.0, 0.0),
            (0.0, 0.0),
            (50.0, 0.0),
            (100.0, 0.0),
        ]
        segments = polyline_segments(points, closed=False)
        filtered = segments_intersecting_aabb(
            segments, -10.0, 10.0, -10.0, 10.0, pad=4.0
        )
        self.assertEqual(len(filtered), 2)
        sample_x: float = 1.0
        sample_z: float = 2.0
        self.assertAlmostEqual(
            distance_to_segment_list(sample_x, sample_z, filtered),
            distance_to_segments(sample_x, sample_z, points, closed=False),
            places=6,
        )

    def test_path_segments_near_endpoint(self) -> None:
        nodes: dict[str, Node] = {
            "shape": Node(
                id="shape",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
            ),
            "a": Node(
                id="a",
                parent_id="shape",
                transform=transform_at(Vector3(-10.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            ),
            "b": Node(
                id="b",
                parent_id="shape",
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            ),
            "c": Node(
                id="c",
                parent_id="shape",
                transform=transform_at(Vector3(10.0, 0.0, 0.0)),
                mesh_id=None,
                role=NodeRole.path_point,
                path_point=default_path_point_recipe(),
            ),
        }
        self.assertEqual(
            path_segments_near_point_ids(
                nodes, "shape", {"a"}, closed=False
            ),
            {0},
        )
        self.assertEqual(
            path_segments_near_point_ids(
                nodes, "shape", {"b"}, closed=False
            ),
            {0, 1},
        )


class TestDeferredGpuFlush(unittest.TestCase):
    def test_defer_holds_revision_until_flush(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("defer")
        catalog.buffers[heightmap_id] = heightmap_buffer_from_samples(
            2, 2, [0.0, 0.0, 0.0, 0.0]
        )
        touch_heightmap_buffer(
            catalog, heightmap_id, dirty_texels=(0, 1, 0, 1), defer_gpu=True
        )
        self.assertEqual(catalog.paint_revisions.get(heightmap_id, 0), 0)
        self.assertIn(heightmap_id, catalog.deferred_paint_bumps)
        self.assertFalse(tick_deferred_gpu_flush(catalog))
        self.assertEqual(catalog.paint_revisions.get(heightmap_id, 0), 0)
        self.assertTrue(tick_deferred_gpu_flush(catalog))
        self.assertEqual(catalog.paint_revisions[heightmap_id], 1)
        self.assertNotIn(heightmap_id, catalog.deferred_paint_bumps)

    def test_immediate_flush_helper(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("flush")
        catalog.buffers[heightmap_id] = heightmap_buffer_from_samples(
            2, 2, [0.0] * 4
        )
        touch_heightmap_buffer(catalog, heightmap_id, defer_gpu=True)
        self.assertTrue(flush_deferred_paint_touches(catalog))
        self.assertEqual(catalog.paint_revisions[heightmap_id], 1)


if __name__ == "__main__":
    unittest.main()
