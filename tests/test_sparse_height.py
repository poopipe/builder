"""sparse height raster paging and clear helpers"""

from __future__ import annotations

import unittest

from builder.heightfield.sparse_height import (
    SparseHeightRaster,
    clear_raster,
    clear_raster_region,
    dense_from_sparse,
    dense_region_from_sparse,
    get_sample,
    set_sample,
    sparse_from_dense,
)


class TestSparseHeight(unittest.TestCase):
    def test_empty_reads_zero(self) -> None:
        raster: SparseHeightRaster = SparseHeightRaster(width=512, height=512)
        self.assertEqual(len(raster.tiles), 0)
        self.assertEqual(get_sample(raster, 0, 0), 0.0)
        self.assertEqual(get_sample(raster, 300, 300), 0.0)

    def test_set_allocates_one_tile(self) -> None:
        raster: SparseHeightRaster = SparseHeightRaster(width=512, height=512)
        set_sample(raster, 10, 20, 3.5)
        self.assertEqual(len(raster.tiles), 1)
        self.assertEqual(get_sample(raster, 10, 20), 3.5)
        self.assertEqual(get_sample(raster, 0, 0), 0.0)

    def test_round_trip_dense(self) -> None:
        dense: list[float] = [0.0, 1.0, 0.0, 2.0]
        raster = sparse_from_dense(2, 2, dense)
        self.assertEqual(dense_from_sparse(raster), dense)

    def test_clear_region_drops_empty_tile(self) -> None:
        raster: SparseHeightRaster = SparseHeightRaster(width=256, height=256)
        set_sample(raster, 1, 1, 4.0)
        self.assertEqual(len(raster.tiles), 1)
        clear_raster_region(raster, 0, 8, 0, 8)
        self.assertEqual(len(raster.tiles), 0)

    def test_clear_all(self) -> None:
        raster = sparse_from_dense(4, 4, [1.0] * 16)
        self.assertGreater(len(raster.tiles), 0)
        clear_raster(raster)
        self.assertEqual(len(raster.tiles), 0)
        self.assertEqual(get_sample(raster, 2, 2), 0.0)

    def test_dense_region(self) -> None:
        raster = sparse_from_dense(4, 4, [float(i) for i in range(16)])
        region = dense_region_from_sparse(raster, 1, 3, 1, 3)
        self.assertEqual(region, [5.0, 6.0, 9.0, 10.0])

    def test_sample_bounds_ignore_tile_padding(self) -> None:
        raster = sparse_from_dense(2, 2, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(raster.sample_min, 1.0)
        self.assertEqual(raster.sample_max, 4.0)


if __name__ == "__main__":
    unittest.main()
