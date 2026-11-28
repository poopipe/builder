"""sculpt brush stamp and paint buffer helpers"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.heightfield.brushes.add import add_stamp, AddBrushParams
from builder.heightfield.brushes.brush_types import BrushStrokeContext
from builder.heightfield.brushes.flatten import (
    FlattenBrushParams,
    flatten_down_stamp,
    flatten_stamp,
    flatten_up_stamp,
)
from builder.heightfield.brushes.smooth import smooth_stamp, SmoothBrushParams
from builder.heightfield.brushes.subtract import subtract_stamp, SubtractBrushParams
from builder.heightfield.heightmap_catalog import (
    HeightmapCatalog,
    heightmap_buffer_from_samples,
    materialize_heightmap_samples,
)
from builder.heightfield.heightfield_types import Heightfield, PaintHeightSource
from builder.heightfield.paint import (
    append_paint_edit_layer,
    create_paint_heightmap,
    paint_revision,
    stamp_paint_layer,
)
from builder.heightfield.sparse_height import get_sample, set_sample


class TestBrushFalloffStamp(unittest.TestCase):
    def test_add_raises_center(self) -> None:
        width: int = 8
        height: int = 8
        raster = heightmap_buffer_from_samples(
            width, height, [0.0] * (width * height)
        )
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=0.25,
            radius_v=0.25,
            intensity=1.0,
        )
        add_stamp(raster, width, height, context, AddBrushParams())
        self.assertGreater(get_sample(raster, width // 2, height // 2), 0.5)
        self.assertAlmostEqual(get_sample(raster, 0, 0), 0.0)

    def test_subtract_lowers_center(self) -> None:
        width: int = 8
        height: int = 8
        raster = heightmap_buffer_from_samples(
            width, height, [1.0] * (width * height)
        )
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=0.25,
            radius_v=0.25,
            intensity=1.0,
        )
        subtract_stamp(raster, width, height, context, SubtractBrushParams())
        self.assertLess(get_sample(raster, width // 2, height // 2), 0.5)

    def test_flatten_pulls_toward_target(self) -> None:
        width: int = 4
        height: int = 4
        raster = heightmap_buffer_from_samples(
            width,
            height,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                0.0,
                0.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=0.6,
            radius_v=0.6,
            intensity=1.0,
            flatten_target=0.5,
        )
        flatten_stamp(raster, width, height, context, FlattenBrushParams())
        # soft falloff at this texel; expect a clear pull toward 0.5 from 2.0
        self.assertLess(get_sample(raster, 1, 1), 2.0)
        self.assertGreater(get_sample(raster, 1, 1), 0.5)

    def test_flatten_up_only_raises(self) -> None:
        width: int = 4
        height: int = 1
        raster = heightmap_buffer_from_samples(
            width, height, [0.0, 0.2, 0.8, 1.0]
        )
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=1.0,
            radius_v=1.0,
            intensity=1.0,
            flatten_target=0.5,
        )
        flatten_up_stamp(raster, width, height, context, FlattenBrushParams())
        self.assertGreater(get_sample(raster, 0, 0), 0.0)
        self.assertGreater(get_sample(raster, 1, 0), 0.2)
        self.assertAlmostEqual(get_sample(raster, 2, 0), 0.8)
        self.assertAlmostEqual(get_sample(raster, 3, 0), 1.0)

    def test_flatten_down_only_lowers(self) -> None:
        width: int = 4
        height: int = 1
        raster = heightmap_buffer_from_samples(
            width, height, [0.0, 0.2, 0.8, 1.0]
        )
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=1.0,
            radius_v=1.0,
            intensity=1.0,
            flatten_target=0.5,
        )
        flatten_down_stamp(raster, width, height, context, FlattenBrushParams())
        self.assertAlmostEqual(get_sample(raster, 0, 0), 0.0)
        self.assertAlmostEqual(get_sample(raster, 1, 0), 0.2)
        self.assertLess(get_sample(raster, 2, 0), 0.8)
        self.assertLess(get_sample(raster, 3, 0), 1.0)

    def test_smooth_reduces_spike(self) -> None:
        width: int = 5
        height: int = 5
        raster = heightmap_buffer_from_samples(
            width, height, [0.0] * (width * height)
        )
        set_sample(raster, width // 2, height // 2, 1.0)
        context: BrushStrokeContext = BrushStrokeContext(
            center_u=0.5,
            center_v=0.5,
            radius_u=0.6,
            radius_v=0.6,
            intensity=1.0,
        )
        smooth_stamp(raster, width, height, context, SmoothBrushParams())
        center: float = get_sample(raster, width // 2, height // 2)
        self.assertLess(center, 1.0)
        self.assertGreater(center, 0.0)


class TestPaintLayer(unittest.TestCase):
    def test_create_and_stamp(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir: Path = Path(tmp)
            heightmap_id = create_paint_heightmap(
                catalog, label="Paint", width=16, height=16, cache_dir=cache_dir
            )
            context: BrushStrokeContext = BrushStrokeContext(
                center_u=0.5,
                center_v=0.5,
                radius_u=0.3,
                radius_v=0.3,
                intensity=2.0,
            )
            stamped: bool = stamp_paint_layer(
                catalog, heightmap_id, "add", AddBrushParams(), context
            )
            self.assertTrue(stamped)
            buffer = catalog.buffers[heightmap_id]
            samples = materialize_heightmap_samples(buffer)
            self.assertGreater(max(samples), 0.0)
            self.assertLessEqual(max(samples), 1.0)
            self.assertGreaterEqual(min(samples), -1.0)
            self.assertEqual(catalog.paint_revisions[heightmap_id], 1)

    def test_paint_default_fill_is_zero(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightmap_id = create_paint_heightmap(
                catalog, label="Paint", width=4, height=4, cache_dir=Path(tmp)
            )
            buffer = catalog.buffers[heightmap_id]
            self.assertEqual(len(buffer.tiles), 0)
            self.assertTrue(
                all(
                    sample == 0.0
                    for sample in materialize_heightmap_samples(buffer)
                )
            )

    def test_stamp_mutates_same_buffer_and_bumps_revision(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightmap_id = create_paint_heightmap(
                catalog, label="Paint", width=8, height=8, cache_dir=Path(tmp)
            )
            before = catalog.buffers[heightmap_id]
            tiles_id: int = id(before.tiles)
            rev0: int = paint_revision(catalog, heightmap_id)
            context: BrushStrokeContext = BrushStrokeContext(
                center_u=0.5,
                center_v=0.5,
                radius_u=0.4,
                radius_v=0.4,
                intensity=1.0,
            )
            self.assertTrue(
                stamp_paint_layer(
                    catalog, heightmap_id, "add", AddBrushParams(), context
                )
            )
            self.assertTrue(
                stamp_paint_layer(
                    catalog, heightmap_id, "add", AddBrushParams(), context
                )
            )
            after = catalog.buffers[heightmap_id]
            self.assertIs(after, before)
            self.assertEqual(id(after.tiles), tiles_id)
            self.assertEqual(paint_revision(catalog, heightmap_id), rev0 + 2)

    def test_append_paint_edit_layer(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightfield: Heightfield = Heightfield()
        with tempfile.TemporaryDirectory() as tmp:
            updated = append_paint_edit_layer(
                catalog, heightfield, cache_dir=Path(tmp)
            )
            self.assertEqual(len(updated.edit_layers), 1)
            self.assertIsInstance(updated.edit_layers[0].height, PaintHeightSource)


if __name__ == "__main__":
    unittest.main()
