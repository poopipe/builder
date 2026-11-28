"""heightmap decode, raw cache, and sample conversion"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from builder.heightfield.height_image_decode import (
    samples_from_u8,
    samples_from_u16,
)
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.height_raw_cache import (
    read_height_raw_cache,
    write_height_raw_cache,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapCatalog,
    HeightmapId,
    import_heightmap_image,
    materialize_heightmap_samples,
)
from builder.heightfield.pillow_height_decode import pillow_decode_height_image


class TestSamplesFromU8(unittest.TestCase):
    def test_endpoints(self) -> None:
        samples: list[float] = samples_from_u8(bytes([0, 255]))
        self.assertEqual(samples, [0.0, 1.0])

    def test_backslash_byte_is_numeric(self) -> None:
        # ffi.buffer[i] yields b'\\' for value 92; float(b'\\') raised ValueError
        raw: bytes = bytes([92])
        samples: list[float] = samples_from_u8(raw)
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0], 92.0 / 255.0)

    def test_rejects_indexing_buffer_slice_as_float(self) -> None:
        slice_byte: bytes = bytes([92])[0:1]
        self.assertIsInstance(slice_byte, bytes)
        with self.assertRaises(ValueError):
            float(slice_byte)


class TestSamplesFromU16(unittest.TestCase):
    def test_endpoints(self) -> None:
        import struct

        packed: bytes = struct.pack("<2H", 0, 65535)
        samples: list[float] = samples_from_u16(packed)
        self.assertEqual(samples, [0.0, 1.0])


class TestPillowU16AndRawCache(unittest.TestCase):
    def test_import_caches_u16_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root: Path = Path(temp_dir)
            png_path: Path = root / "height_u16.png"
            image: Image.Image = Image.new("I;16", (2, 2))
            image.putpixel((0, 0), 0)
            image.putpixel((1, 0), 65535)
            image.putpixel((0, 1), 32768)
            image.putpixel((1, 1), 1)
            image.save(png_path)

            catalog: HeightmapCatalog = HeightmapCatalog()
            asset = import_heightmap_image(
                catalog,
                png_path,
                HeightmapId("height_u16"),
                "height_u16",
            )
            self.assertEqual(asset.sample_type, HeightSampleType.u16)
            self.assertEqual(asset.width, 2)
            self.assertEqual(asset.height, 2)
            cache_path: Path = Path(asset.cache_path)
            self.assertTrue(cache_path.is_file())

            decoded = pillow_decode_height_image(png_path)
            cached = read_height_raw_cache(cache_path)
            self.assertEqual(cached.sample_type, HeightSampleType.u16)
            self.assertEqual(cached.width, decoded.width)
            self.assertEqual(cached.height, decoded.height)
            self.assertEqual(cached.data, decoded.data)

            write_height_raw_cache(cache_path, decoded)
            again = read_height_raw_cache(cache_path)
            self.assertEqual(again.data, decoded.data)

            buffer = catalog.buffers[HeightmapId("height_u16")]
            samples = materialize_heightmap_samples(buffer)
            self.assertEqual(samples[0], 0.0)
            self.assertEqual(samples[1], 1.0)
            self.assertAlmostEqual(samples[2], 32768.0 / 65535.0)
            self.assertAlmostEqual(samples[3], 1.0 / 65535.0)


if __name__ == "__main__":
    unittest.main()
