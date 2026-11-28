"""pending paint dirty rect tracking for gpu uploads"""

from __future__ import annotations

import unittest
from dataclasses import replace

from builder.generators.param_types import FloatParam
from builder.heightfield.gpu.heightfield_gpu import pending_dirty_world_aabb
from builder.heightfield.heightfield_types import (
    Heightfield,
    default_heightfield,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightSampleType,
    HeightmapAsset,
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
    register_heightmap_asset,
)
from builder.heightfield.paint import record_paint_dirty, touch_heightmap_buffer
from builder.heightfield.vector.stamp import (
    texel_range_to_world_aabb,
    world_aabb_to_texel_range,
)


class TestRecordPaintDirty(unittest.TestCase):
    def test_unions_partial_rects(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("paint")
        record_paint_dirty(catalog, heightmap_id, (1, 4, 2, 5))
        record_paint_dirty(catalog, heightmap_id, (3, 8, 0, 3))
        self.assertEqual(catalog.paint_dirty_rects[heightmap_id], (1, 8, 0, 5))

    def test_full_dirty_wins(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("paint")
        record_paint_dirty(catalog, heightmap_id, (1, 4, 2, 5))
        record_paint_dirty(catalog, heightmap_id, None)
        self.assertIsNone(catalog.paint_dirty_rects[heightmap_id])
        record_paint_dirty(catalog, heightmap_id, (0, 1, 0, 1))
        self.assertIsNone(catalog.paint_dirty_rects[heightmap_id])

    def test_touch_records_dirty(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("paint")
        catalog.buffers[heightmap_id] = heightmap_buffer_from_samples(
            4, 4, [0.0] * 16
        )
        touch_heightmap_buffer(
            catalog, heightmap_id, dirty_texels=(1, 3, 1, 3)
        )
        self.assertEqual(catalog.paint_revisions[heightmap_id], 1)
        self.assertEqual(catalog.paint_dirty_rects[heightmap_id], (1, 3, 1, 3))


class TestTexelWorldRoundTrip(unittest.TestCase):
    def test_world_covers_texel_range(self) -> None:
        size_x: float = 256.0
        size_z: float = 256.0
        width: int = 64
        height: int = 64
        world = texel_range_to_world_aabb(
            8, 24, 10, 30, size_x, size_z, width, height
        )
        back = world_aabb_to_texel_range(
            world[0],
            world[1],
            world[2],
            world[3],
            size_x,
            size_z,
            width,
            height,
        )
        self.assertIsNotNone(back)
        assert back is not None
        self.assertLessEqual(back[0], 8)
        self.assertGreaterEqual(back[1], 24)
        self.assertLessEqual(back[2], 10)
        self.assertGreaterEqual(back[3], 30)


class TestPendingDirtyWorldAabb(unittest.TestCase):
    def test_partial_pending_maps_to_world(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        base_id: HeightmapId = HeightmapId("base")
        register_heightmap_asset(
            catalog,
            HeightmapAsset(
                heightmap_id=base_id,
                label="base",
                source_path="base.f32",
                cache_path="base.hraw",
                sample_type=HeightSampleType.f32,
                width=64,
                height=64,
            ),
        )
        catalog.buffers[base_id] = heightmap_buffer_from_samples(
            64, 64, [0.0] * (64 * 64)
        )
        heightfield: Heightfield = replace(
            default_heightfield(),
            base=base_id,
            size_x=FloatParam(256.0, label="Size X"),
            size_z=FloatParam(256.0, label="Size Z"),
        )
        layout = heightfield_layout(heightfield)
        record_paint_dirty(catalog, base_id, (0, 8, 0, 8))
        dirty = pending_dirty_world_aabb(catalog, heightfield, layout)
        self.assertIsNotNone(dirty)
        assert dirty is not None
        self.assertLess(dirty[0], dirty[1])
        self.assertLess(dirty[2], dirty[3])


if __name__ == "__main__":
    unittest.main()
