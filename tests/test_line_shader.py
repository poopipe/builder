"""unit tests for screen-space line batch sizing"""

from __future__ import annotations

import unittest

from pyray import Color

from builder.view.line_shader import (
    LineBatch,
    LineMeshCache,
    add_line_segment,
    clear_line_batch,
    line_batch_buffer_counts,
    sync_line_mesh_cache,
)


class LineBatchBufferCountsTests(unittest.TestCase):
    def test_zero_segments(self) -> None:
        verts: int
        indices: int
        verts, indices = line_batch_buffer_counts(0)
        self.assertEqual(verts, 0)
        self.assertEqual(indices, 0)

    def test_n_segments(self) -> None:
        verts: int
        indices: int
        verts, indices = line_batch_buffer_counts(3)
        self.assertEqual(verts, 12)
        self.assertEqual(indices, 18)

    def test_negative_treated_as_zero(self) -> None:
        verts: int
        indices: int
        verts, indices = line_batch_buffer_counts(-2)
        self.assertEqual(verts, 0)
        self.assertEqual(indices, 0)


class LineBatchAccumulateTests(unittest.TestCase):
    def test_add_and_clear(self) -> None:
        batch: LineBatch = LineBatch()
        add_line_segment(
            batch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, Color(255, 0, 0, 255), 3.0
        )
        add_line_segment(
            batch, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, Color(0, 255, 0, 255), 3.0
        )
        self.assertEqual(len(batch.segments), 2)
        verts: int
        indices: int
        verts, indices = line_batch_buffer_counts(len(batch.segments))
        self.assertEqual(verts, 8)
        self.assertEqual(indices, 12)
        clear_line_batch(batch)
        self.assertEqual(len(batch.segments), 0)

    def test_skips_degenerate(self) -> None:
        batch: LineBatch = LineBatch()
        add_line_segment(
            batch, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, Color(255, 255, 255, 255), 3.0
        )
        self.assertEqual(len(batch.segments), 0)

    def test_skips_non_positive_thickness(self) -> None:
        batch: LineBatch = LineBatch()
        add_line_segment(
            batch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, Color(255, 255, 255, 255), 0.0
        )
        self.assertEqual(len(batch.segments), 0)


class LineMeshCacheSyncTests(unittest.TestCase):
    def test_matching_key_skips_rebuild_and_clears_batch(self) -> None:
        cache: LineMeshCache = LineMeshCache()
        cache.key = ("same",)
        batch: LineBatch = LineBatch()
        add_line_segment(
            batch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, Color(255, 0, 0, 255), 3.0
        )
        sync_line_mesh_cache(cache, ("same",), batch)
        self.assertEqual(len(batch.segments), 0)
        self.assertEqual(cache.key, ("same",))
        self.assertEqual(len(cache.meshes), 0)


if __name__ == "__main__":
    unittest.main()
