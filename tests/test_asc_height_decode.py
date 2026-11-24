"""ESRI .asc height decode, optional meta, and base import hints"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.commands.file import apply_asc_base_import_hints
from builder.heightfield.asc_height_decode import (
    asc_decode_height_image,
    asc_import_hints,
    load_asc_height_meta,
)
from builder.heightfield.height_image_decode import (
    default_decode_height_image,
    samples_from_f32,
)
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.heightfield_types import Heightfield, default_heightfield
from builder.heightfield.heightmap_catalog import (
    HeightmapCatalog,
    HeightmapId,
    import_heightmap_image,
)


def write_tiny_asc(path: Path, *, nodata: float = -9999.0) -> None:
    """2x2 north-first grid: row0=[1,2] north, row1=[3,NODATA] south"""
    path.write_text(
        "\n".join(
            [
                "ncols         2",
                "nrows         2",
                "xllcorner     0.0",
                "yllcorner     0.0",
                "cellsize      10.0",
                f"NODATA_value  {nodata}",
                "1.0 2.0",
                f"3.0 {nodata}",
                "",
            ]
        ),
        encoding="utf-8",
    )


class TestAscDecode(unittest.TestCase):
    def test_f32_row_flip_and_nodata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            decoded = asc_decode_height_image(asc_path)
            self.assertEqual(decoded.width, 2)
            self.assertEqual(decoded.height, 2)
            self.assertEqual(decoded.sample_type, HeightSampleType.f32)
            samples: list[float] = samples_from_f32(decoded.data)
            # south-first, columns mirrored: [0, 3], then north [2, 1]
            self.assertEqual(samples, [0.0, 3.0, 2.0, 1.0])

    def test_default_decode_dispatches_asc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            decoded = default_decode_height_image(asc_path)
            self.assertEqual(decoded.sample_type, HeightSampleType.f32)

    def test_import_warms_f32_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            catalog: HeightmapCatalog = HeightmapCatalog()
            asset = import_heightmap_image(
                catalog, asc_path, HeightmapId("height"), "height"
            )
            self.assertEqual(asset.sample_type, HeightSampleType.f32)
            buffer = catalog.buffers[HeightmapId("height")]
            self.assertEqual(buffer.samples, [0.0, 3.0, 2.0, 1.0])


class TestAscMetaAndHints(unittest.TestCase):
    def test_meta_missing_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            self.assertIsNone(load_asc_height_meta(asc_path))
            hints = asc_import_hints(asc_path)
            self.assertEqual(hints.size_x, 20.0)
            self.assertEqual(hints.size_z, 20.0)
            self.assertEqual(hints.origin_x, 0.0)
            self.assertEqual(hints.origin_z, 0.0)
            self.assertIsNone(hints.height_min)
            self.assertIsNone(hints.height_max)

    def test_meta_elevation_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            meta_path: Path = Path(temp_dir) / "height_meta.json"
            meta_path.write_text(
                '{"elevation_m": {"min": -10.5, "max": 42.0}}',
                encoding="utf-8",
            )
            meta = load_asc_height_meta(asc_path)
            assert meta is not None
            self.assertAlmostEqual(meta.height_min, -10.5)
            self.assertAlmostEqual(meta.height_max, 42.0)
            hints = asc_import_hints(asc_path)
            self.assertAlmostEqual(hints.height_min or 0.0, -10.5)
            self.assertAlmostEqual(hints.height_max or 0.0, 42.0)

    def test_base_hints_set_size_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "terrain.asc"
            # header-only is enough for hints; 256x256 at 1m snaps cleanly
            asc_path.write_text(
                "\n".join(
                    [
                        "ncols         256",
                        "nrows         256",
                        "xllcorner     200000.0",
                        "yllcorner     1000.0",
                        "cellsize      1.0",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            meta_path: Path = Path(temp_dir) / "terrain_meta.json"
            meta_path.write_text(
                '{"elevation_m": {"min": -5.0, "max": 100.0}}',
                encoding="utf-8",
            )
            heightfield: Heightfield = default_heightfield()
            hints = asc_import_hints(asc_path)
            updated = apply_asc_base_import_hints(heightfield, hints)
            self.assertEqual(updated.size_x.value, 256.0)
            self.assertEqual(updated.size_z.value, 256.0)
            self.assertEqual(updated.origin_x.value, 200000.0)
            self.assertEqual(updated.origin_z.value, 1000.0)
            self.assertEqual(updated.base_height_min.value, -5.0)
            self.assertEqual(updated.base_height_max.value, 100.0)

    def test_base_hints_without_meta_keep_height_bounds(self) -> None:
        heightfield: Heightfield = default_heightfield()
        original_min: float = heightfield.base_height_min.value
        original_max: float = heightfield.base_height_max.value
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "height.asc"
            write_tiny_asc(asc_path)
            updated = apply_asc_base_import_hints(
                heightfield, asc_import_hints(asc_path)
            )
            # 20m snaps to 256m quantum
            self.assertEqual(updated.size_x.value, 256.0)
            self.assertEqual(updated.size_z.value, 256.0)
            self.assertEqual(updated.base_height_min.value, original_min)
            self.assertEqual(updated.base_height_max.value, original_max)


if __name__ == "__main__":
    unittest.main()
